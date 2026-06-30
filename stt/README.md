# `stt/` — voice/text-directed targeting

Describe who to track (typed or spoken) → a **multimodal LLM** finds that person
in the current frame and returns their bounding box → that box seeds the tracker.

Ported from `wardmate_ws` (STT + LLM, originally driving a robot arm) and
repointed to person targeting.

## Pieces
| file | what |
|---|---|
| `speech_to_text.py` | `SpeechToText`: mic → energy VAD → Silero VAD → faster-whisper → text (zh/en). Ported verbatim. |
| `target_resolver.py` | `resolve_target(frame_rgb, description) -> [x1,y1,x2,y2]`: multimodal LLM grounds the description to a person bbox. |
| `app.py` | end-to-end CLI: frame + (typed/spoken) description → bbox drawn on an image. |

## Install
```bash
source .venv/bin/activate
pip install -r stt/requirements.txt
# voice only: sudo apt-get install -y portaudio19-dev   (needed by pyaudio)
```

## Backend
Default is **local Qwen2-VL-2B-Instruct** via transformers — no API key, no
server, runs on the GPU (`STT_BACKEND=hf`). The model is pulled from the HF cache
on first use. Verified: resolves English and Chinese descriptions to a person
bbox in ~2-3 s/query (after a one-time ~30 s load).

To use a hosted/OpenAI-compatible vision endpoint instead:
```bash
export STT_BACKEND=openai
export LLM_MODEL=gpt-4o-mini LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=sk-...
# or local Ollama vision:  LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5vl:7b LLM_API_KEY=ollama
```

## Run
```bash
# typed description on a video frame
python -m stt.app --video videos/798511637.725509.mp4 --frame 0 \
    --text "the person in the red shirt"

# speak the description (USB mic)
python -m stt.app --video videos/798511637.725509.mp4 --frame 0 --voice
```
Prints `bbox = [x1,y1,x2,y2]` and writes `stt_target.jpg` with the box drawn.

## Hooking into the tracker
The returned bbox is exactly the seed `run_video.py` expects (it currently gets
its seed from `identify_person_by_color`). Next step is letting `run_video`
take `--describe "<text>"` and seed the AOT / sam2-aotmem tracker from
`resolve_target` instead of the colour heuristic.
