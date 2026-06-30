"""Multimodal target resolver: Chinese description -> target person.

Two ways to ground a description to a person:
  * select()  (recommended) — YOLO detects the people, we draw numbered boxes,
    and the VLM picks WHICH number matches the description. The returned box is
    YOLO's (precise), and "pick a number" is a task small VLMs do reliably.
  * resolve() — ask the VLM directly for a normalised bbox (no detector). Looser.

Backend (env STT_BACKEND): "hf" (default) = local Qwen2-VL-2B-Instruct via
transformers (no API key); "openai" = any OpenAI-compatible vision endpoint
(LLM_MODEL / LLM_BASE_URL / LLM_API_KEY).
"""
import base64
import json
import os
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --- openai backend config (only used when backend="openai") ---
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
API_KEY = (os.environ.get("LLM_API_KEY")
           or os.environ.get("OPENAI_API_KEY") or "ollama")

# --- hf backend config ---
HF_MODEL_ID = os.environ.get("STT_VLM", "Qwen/Qwen2-VL-2B-Instruct")
BACKEND = os.environ.get("STT_BACKEND", "hf")

# Prompt for the recommended path: pick a numbered detection.
_SELECT_SYSTEM = (
    "你會看到一張相機畫面，畫面中每個人都被標上一個綠色編號方框（0、1、2…）。"
    "根據使用者的中文描述，選出最符合的那「一個」人。\n"
    '只回傳一個 JSON 物件，不要任何其他文字：{"id": 編號}。'
    '若沒有任何人符合描述，回傳 {"id": -1}。\n'
    "請依衣服顏色、位置（左／右／中）、遠近判斷。"
)
# Prompt for the detector-free path: ask for a normalised bbox.
_BOX_SYSTEM = (
    "你是人物追蹤機器人的目標選定模組。你會看到一張相機畫面，以及一段「要追蹤哪一個人」"
    "的中文描述。請找出那「一個」人，回傳他的邊界框。\n"
    "只回傳一個 JSON 物件，不要任何其他文字：\n"
    '  {"found": true, "box": [x1, y1, x2, y2]}\n'
    "其中 box 為正規化座標 0 到 1（x = 寬度比例，y = 高度比例），需滿足 x1<x2 且 "
    "y1<y2，並緊貼整個人。\n"
    '若畫面中沒有人符合描述，回傳 {"found": false, "box": null}。'
)
_USER_PREFIX = "追蹤這個人："


def _encode_jpeg(image_rgb: np.ndarray, max_width: int = 1024) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(h * max_width / w)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf).decode("ascii")


def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_box(reply: str, w: int, h: int) -> Optional[List[int]]:
    """Reply -> pixel bbox in (w, h). Handles 0..1, Qwen 0..1000, or pixels."""
    data = _extract_json(reply)
    if not data or not data.get("found") or not data.get("box"):
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in data["box"])
    except (ValueError, TypeError):
        return None
    mx = max(x1, y1, x2, y2)
    if mx <= 1.5:
        x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
    elif mx <= 1000.5:
        x1, x2 = x1 / 1000 * w, x2 / 1000 * w
        y1, y2 = y1 / 1000 * h, y2 / 1000 * h
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def _annotate(image_rgb: np.ndarray, boxes) -> np.ndarray:
    """Draw numbered green boxes on a copy of the frame (for the VLM to pick)."""
    vis = np.ascontiguousarray(image_rgb).copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        lbl = str(i)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 12)),
                      (x1 + tw + 12, y1), (0, 255, 0), -1)
        cv2.putText(vis, lbl, (x1 + 6, max(th, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return vis


# Appearance / role words — if the description has any of these it isn't a
# *pure* spatial pick, so we send it to the VLM (which reads colour/role) even
# if it also mentions a direction.
_APPEARANCE = ("紅", "橙", "黃", "綠", "藍", "紫", "白", "黑", "灰", "粉", "棕",
               "醫生", "醫師", "護士", "護理", "病人", "衣", "袍", "褲", "裙",
               "帽", "眼鏡", "口罩", "頭髮", "鬍", "男", "女", "小孩", "老", "胖", "瘦")


def _spatial_pick(boxes, desc) -> Optional[int]:
    """Resolve a PURELY positional/size description to a box index from YOLO
    geometry (exact). Returns None if it mentions appearance/role (VLM job) or
    has no spatial word."""
    d = desc.strip()
    if any(a in d for a in _APPEARANCE):
        return None
    cxs = [(x1 + x2) / 2 for x1, y1, x2, y2 in boxes]
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    if any(k in d for k in ("最右", "右邊", "右側", "右方", "右手邊")):
        return int(max(range(len(boxes)), key=lambda i: cxs[i]))
    if any(k in d for k in ("最左", "左邊", "左側", "左方", "左手邊")):
        return int(min(range(len(boxes)), key=lambda i: cxs[i]))
    if any(k in d for k in ("最近", "最前", "前面", "最大", "最大隻")):
        return int(max(range(len(boxes)), key=lambda i: areas[i]))
    if any(k in d for k in ("最遠", "最後", "後面", "最小", "最小隻")):
        return int(min(range(len(boxes)), key=lambda i: areas[i]))
    return None


def _spatial_bonus(boxes, desc, weight=3.0) -> List[float]:
    """Per-box additive bonus from YOLO geometry for any directional/size word
    in the description (all zeros if there is none). Combined with the VLM's
    appearance score so 「左邊穿白色的女生」 resolves correctly."""
    n = len(boxes)
    bonus = [0.0] * n
    cxs = [(x1 + x2) / 2 for x1, y1, x2, y2 in boxes]
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    if any(k in desc for k in ("最右", "右邊", "右側", "右方", "右手邊")):
        bonus[max(range(n), key=lambda i: cxs[i])] += weight
    if any(k in desc for k in ("最左", "左邊", "左側", "左方", "左手邊")):
        bonus[min(range(n), key=lambda i: cxs[i])] += weight
    if any(k in desc for k in ("最近", "最前", "前面", "最大")):
        bonus[max(range(n), key=lambda i: areas[i])] += weight
    if any(k in desc for k in ("最遠", "最後", "後面", "最小")):
        bonus[min(range(n), key=lambda i: areas[i])] += weight
    if any(k in desc for k in ("中間", "中央", "正中")):
        mid = (min(cxs) + max(cxs)) / 2
        bonus[min(range(n), key=lambda i: abs(cxs[i] - mid))] += weight
    return bonus


class TargetResolver:
    """Resolve a Chinese description to a person (by detection-pick or bbox)."""

    def __init__(self, backend: str = BACKEND, model: str = MODEL,
                 base_url: str = BASE_URL, api_key: str = API_KEY,
                 hf_model_id: str = HF_MODEL_ID, device: str = "cuda"):
        self.backend = backend
        self.device = device
        if backend == "openai":
            from openai import OpenAI
            self.model = model
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        elif backend == "hf":
            self.hf_model_id = hf_model_id
            self._hf_model = None
            self._hf_proc = None
        else:
            raise ValueError(f"unknown backend {backend!r} (use 'hf' or 'openai')")

    # ---- backends: (image, system, user_text) -> raw reply --------------------
    def _reply_openai(self, image_rgb, system, user_text) -> str:
        b64 = _encode_jpeg(image_rgb)
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=300,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ])
        return resp.choices[0].message.content or ""

    def _load_hf(self):
        if self._hf_model is not None:
            return
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self._hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.hf_model_id, dtype=torch.bfloat16, device_map=self.device).eval()
        self._hf_proc = AutoProcessor.from_pretrained(
            self.hf_model_id, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

    def _reply_hf(self, image_rgb, system, user_text) -> str:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        self._load_hf()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image", "image": Image.fromarray(image_rgb)},
                {"type": "text", "text": user_text},
            ]},
        ]
        text = self._hf_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = self._hf_proc(text=[text], images=imgs, videos=vids,
                               padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._hf_model.generate(**inputs, max_new_tokens=128,
                                          do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self._hf_proc.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

    def _reply(self, image_rgb, system, user_text) -> str:
        if self.backend == "hf":
            return self._reply_hf(image_rgb, system, user_text)
        return self._reply_openai(image_rgb, system, user_text)

    # ---- public --------------------------------------------------------------
    def select(self, image_rgb, description, boxes) -> int:
        """Pick which detection (index into ``boxes``, xyxy) matches the Chinese
        description. Returns the index, or -1 if none / no detections.

        Purely spatial words (左/右/中/最近/最遠/最大/最小) are resolved exactly
        from the YOLO box geometry — no VLM, since position can't be read from a
        crop and the small VLM is unreliable at it. Everything else (clothing
        colour, role) goes to the VLM with a SIMPLE direct prompt on the full
        numbered frame (heavy JSON/system prompts make the 2B model default to
        one box)."""
        if not boxes or not description or not description.strip():
            return -1
        geo = _spatial_pick(boxes, description)
        if geo is not None:
            return geo
        return self._select_by_crops(image_rgb, description, boxes)

    def _select_by_crops(self, image_rgb, description, boxes) -> int:
        """Score each detection by (VLM appearance on its crop) + (deterministic
        position bonus from the YOLO geometry).

        We tried handing the crop's coordinates to the VLM (the natural idea),
        but the 2B model doesn't reliably reason over textual position — it even
        regressed the pure-spatial cases. So instead the VLM judges ONLY
        appearance from the crop (which it does well in isolation), and any
        position/distance word in the request adds a geometry-computed bonus to
        the matching box. Both signals combine, so 「左邊穿白色的女生」 works
        even when the crop alone can't convey "left"."""
        bonus = _spatial_bonus(boxes, description, weight=3.0)
        best_idx, best = -1, -1e9
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            crop = image_rgb[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            reply = self._reply(
                crop, "",
                f"這張圖中的人符合「{description}」嗎？請給 0 到 10 分"
                "（10=完全符合，0=完全不符合）。只回答一個數字。")
            m = re.search(r"\d+", reply or "")
            app = float(m.group(0)) if m else 0.0
            score = app + bonus[i]
            if score > best:
                best, best_idx = score, i
        return best_idx

    def resolve(self, image_rgb, description) -> Optional[List[int]]:
        """Detector-free: ask the VLM directly for the person's bbox."""
        if not description or not description.strip():
            return None
        h, w = image_rgb.shape[:2]
        reply = self._reply(image_rgb, _BOX_SYSTEM,
                            f"{_USER_PREFIX}「{description}」")
        return _parse_box(reply, w, h)


def detect_people(image_rgb, yolo_model_path="yolo11m.pt", conf=0.4) -> List[List[int]]:
    """YOLO person boxes (xyxy int) in the frame."""
    from ultralytics import YOLO
    res = YOLO(yolo_model_path)(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                                verbose=False, conf=conf)[0]
    out = []
    for b in res.boxes:
        if int(b.cls) == 0:  # person
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            out.append([int(x1), int(y1), int(x2), int(y2)])
    return out


def detect_and_select(image_rgb, description, yolo_model_path="yolo11m.pt",
                      conf=0.4, resolver: "TargetResolver" = None
                      ) -> Tuple[Optional[List[int]], List[List[int]]]:
    """YOLO-detect people, then have the VLM pick the one matching ``description``.

    Returns (chosen_box_xyxy_or_None, all_person_boxes).
    """
    boxes = detect_people(image_rgb, yolo_model_path, conf)
    if not boxes:
        return None, []
    r = resolver or TargetResolver()
    idx = r.select(image_rgb, description, boxes)
    return (boxes[idx] if idx >= 0 else None), boxes


def resolve_target(image_rgb, description, **kwargs) -> Optional[List[int]]:
    """One-shot detector-free bbox (kept for the simple app path)."""
    return TargetResolver(**kwargs).resolve(image_rgb, description)
