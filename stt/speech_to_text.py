#!/usr/bin/env python3
"""Speech-to-text: mic -> energy VAD -> Silero VAD trim -> faster-whisper -> text.

Ported verbatim from wardmate_ws/src/stt/scripts/main_stt.py. Self-contained
(no ROS): a reusable ``SpeechToText`` that calls ``on_text(text, lang)`` for
every recognised command. Used here to capture the spoken description of who to
track (e.g. 「追蹤穿紅色衣服的人」 / "track the person in the red shirt").
"""
import os
import threading
import time

import numpy as np

# --- 基礎配置（可用環境變數覆寫）---
MODEL_SIZE = os.environ.get("STT_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("STT_DEVICE", "cuda")        # 自動退回 CPU（見 load_model）
COMPUTE_TYPE = os.environ.get("STT_COMPUTE", "int8_float16")

RATE = 16000
CHUNK = 1600                # 100ms
VAD_THRESHOLD = 0.06       # 開始收音的能量門檻，若環境吵雜可調高
SILENCE_DURATION = 0.5
PRE_ROLL_CHUNKS = 6        # 預錄 0.6 秒

# --- 近場 / 信心門檻：只接受「靠近且清楚對著它說」的指令 ---
PROXIMITY_RMS = 0.5        # 整段語音的峰值音量需超過此值，過濾遠處/旁人說話
MIN_SPEECH_DURATION = 0.15  # 有效語音最短秒數
MAX_NO_SPEECH_PROB = 0.9    # Whisper 判為「非語音」的機率上限
MIN_AVG_LOGPROB = -1.3      # Whisper 平均對數機率下限（單字無上下文，放寬）

# 僅限中文，避免雜訊被誤判成其他語言
ALLOWED_LANGUAGES = {"zh"}


def _make_vad_options():
    from faster_whisper.vad import VadOptions
    return VadOptions(
        threshold=0.5,
        min_speech_duration_ms=120,
        min_silence_duration_ms=300,
        speech_pad_ms=300,
    )


def vad_trim(audio):
    """用 Silero VAD 取出語音片段並串接成乾淨音訊。

    回傳 (trimmed_audio, speech_seconds)；整段都判為非語音則回傳 (None, 0.0)。
    """
    from faster_whisper.vad import get_speech_timestamps, collect_chunks
    segments = get_speech_timestamps(audio, _make_vad_options(), sampling_rate=RATE)
    if not segments:
        return None, 0.0
    chunks, _ = collect_chunks(audio, segments, sampling_rate=RATE)
    trimmed = np.concatenate(chunks)
    return trimmed, len(trimmed) / RATE


def detect_allowed_language(model, audio):
    """在 ALLOWED_LANGUAGES 中挑選機率最高的語言（音訊已先經 VAD 修剪）。"""
    _, _, all_probs = model.detect_language(audio)
    for lang, _prob in all_probs:
        if lang in ALLOWED_LANGUAGES:
            return lang
    return "en"


def get_usb_microphone_index(p):
    """自動尋找 USB 麥克風的索引號；找不到則回傳 None（用預設輸入裝置）。"""
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if "USB" in dev['name'].upper():
            return i
    return None


class SpeechToText:
    """可重複使用的語音辨識器：麥克風 → VAD 修剪 → Whisper → 回呼文字。

    所有狀態訊息走 status_cb，辨識到的指令走 on_text，讓呼叫端決定要印出來、
    寫進畫面、還是丟給 LLM（此專案：丟給多模態 LLM 找出要追蹤的人）。
    """

    def __init__(self, status_cb=None, show_rms=False, to_traditional=True):
        self.status_cb = status_cb or (lambda _m: None)
        self.show_rms = show_rms
        self.to_traditional = to_traditional
        self.model = None
        self._s2tw = None
        self._stop = threading.Event()

    def _status(self, msg):
        self.status_cb(msg)

    def load_model(self):
        from faster_whisper import WhisperModel
        self._status(f"載入 Faster-Whisper 模型 ({MODEL_SIZE}, {DEVICE})…")
        try:
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE,
                                      compute_type=COMPUTE_TYPE)
        except (ValueError, RuntimeError) as e:
            # 此平台的 ctranslate2 wheel 未含 CUDA（aarch64/cu13），退回 CPU。
            # 短指令在 CPU 上仍可接受（large-v3-turbo 約數秒）。
            self._status(f"CUDA STT 不可用（{e}）；改用 CPU int8。")
            self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        if self.to_traditional:
            try:
                from opencc import OpenCC
                self._s2tw = OpenCC("s2twp")   # 簡體 → 繁體（台灣慣用詞）
            except Exception as e:                              # noqa: BLE001
                self._status(f"OpenCC 不可用（{e}）；輸出維持簡體。")
                self._s2tw = None
        self._status("模型載入完成。")

    def stop(self):
        self._stop.set()

    def run(self, on_text):
        """阻塞式辨識迴圈；每辨識出一句就呼叫 on_text(text, lang)，直到 stop()。"""
        import pyaudio
        if self.model is None:
            self.load_model()

        p = pyaudio.PyAudio()
        input_idx = get_usb_microphone_index(p)
        if input_idx is not None:
            self._status(f"已自動選定麥克風索引: {input_idx}")

        try:
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                input_device_index=input_idx, frames_per_buffer=CHUNK,
            )
        except Exception as e:                                  # noqa: BLE001
            self._status(f"無法開啟麥克風: {e}")
            p.terminate()
            return

        pre_roll_buffer = []
        audio_buffer = []
        is_speaking = False
        silence_start = None
        peak_rms = 0.0
        self._status(">>> 系統就緒，請說話…")

        try:
            while not self._stop.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_float32 ** 2))

                if self.show_rms:
                    status = " [!] 偵測到聲音" if rms > VAD_THRESHOLD else " [-] 靜音中"
                    print(f"\r當前音量 RMS: {rms:.4f} {status}", end="", flush=True)

                if rms > VAD_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        peak_rms = 0.0
                        audio_buffer = list(pre_roll_buffer)
                        self._status("[收音中…]")
                    peak_rms = max(peak_rms, rms)
                    audio_buffer.append(audio_float32)
                    silence_start = None
                else:
                    if is_speaking:
                        audio_buffer.append(audio_float32)
                        if silence_start is None:
                            silence_start = time.time()
                        if time.time() - silence_start > SILENCE_DURATION:
                            self._process_segment(audio_buffer, peak_rms, on_text)
                            audio_buffer = []
                            is_speaking = False
                            silence_start = None
                            peak_rms = 0.0
                    else:
                        pre_roll_buffer.append(audio_float32)
                        if len(pre_roll_buffer) > PRE_ROLL_CHUNKS:
                            pre_roll_buffer.pop(0)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self._status(">>> 已停止聆聽。")

    def _process_segment(self, audio_buffer, peak_rms, on_text):
        input_audio = np.concatenate(audio_buffer)
        duration = len(input_audio) / RATE
        trimmed, speech_dur = vad_trim(input_audio)

        if peak_rms < PROXIMITY_RMS:
            self._status(f">>> 忽略：非近場指令 (峰值 {peak_rms:.3f} < {PROXIMITY_RMS})")
            return
        if trimmed is None or speech_dur < MIN_SPEECH_DURATION:
            self._status(f">>> 忽略：VAD 未偵測到有效語音 ({speech_dur:.1f}s)")
            return

        self._status(f"[辨識中… VAD 修剪 {duration:.1f}s → {speech_dur:.1f}s]")
        lang = detect_allowed_language(self.model, trimmed)
        segments, info = self.model.transcribe(
            trimmed, beam_size=5, language=lang,
            condition_on_previous_text=False, vad_filter=False,
        )
        segs = list(segments)
        good = [
            s.text for s in segs
            if s.no_speech_prob < MAX_NO_SPEECH_PROB and s.avg_logprob > MIN_AVG_LOGPROB
        ]
        text = "".join(good).strip()
        if info.language == "zh" and self._s2tw is not None:
            text = self._s2tw.convert(text)

        if text:
            on_text(text, info.language)
        else:
            self._status(">>> 未偵測到任何文字（或被信心門檻過濾）")


def main():
    stt = SpeechToText(status_cb=print, show_rms=True)
    print(">>> 測試：請說『追蹤穿紅色衣服的人』")
    try:
        stt.run(on_text=lambda text, lang: print(f"\n>>> 辨識結果 [{lang}]: {text}"))
    except KeyboardInterrupt:
        stt.stop()


if __name__ == "__main__":
    main()
