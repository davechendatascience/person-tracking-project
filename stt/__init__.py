"""Voice-directed targeting: STT (faster-whisper) + multimodal-LLM target picker.

Ported/adapted from wardmate_ws (where STT+LLM drove a robot arm). Here it picks
*which person to track*: describe the target (typed or spoken) and a multimodal
LLM returns that person's bbox on the current frame, which seeds the tracker.

  from stt.speech_to_text import SpeechToText      # mic -> text
  from stt.target_resolver import resolve_target   # (frame, description) -> bbox
"""
from stt.target_resolver import TargetResolver, resolve_target  # noqa: F401

__all__ = ["TargetResolver", "resolve_target", "SpeechToText"]


def __getattr__(name):
    # Lazy: importing SpeechToText pulls heavy audio/CUDA deps, so only on use.
    if name == "SpeechToText":
        from stt.speech_to_text import SpeechToText
        return SpeechToText
    raise AttributeError(name)
