"""How the traveler's voice-settings sliders reach the TTS providers.

Two knobs, both chosen because they survive the trip through either provider
and are audible to someone who is not listening for them:

  speed          how fast the guide talks. Directions read at 1.0 are fine in a
                 quiet gate area and too fast in a crowded concourse.
  expressiveness how much the delivery moves. ElevenLabs calls the same axis
                 `stability` and points it the other way, so the mapping is
                 inverted exactly once, here, rather than at each call site.

Clamping lives here too. A websocket client is not a trusted source of floats:
ElevenLabs rejects a speed outside 0.7–1.2 with a 400, which would turn a
stray slider value into a silent agent, so every value is pinned into range on
the way in rather than checked on the way out.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.config import (
    VOICE_EXPRESSIVENESS_DEFAULT,
    VOICE_SPEED_DEFAULT,
    VOICE_SPEED_MAX,
    VOICE_SPEED_MIN,
)


def _clamp(value, low: float, high: float, fallback: float) -> float:
    """Pin `value` into [low, high]; anything unreadable becomes `fallback`."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    # NaN fails both comparisons, so test for it rather than letting it through.
    if number != number:
        return fallback
    return min(high, max(low, number))


@dataclass(frozen=True)
class VoiceTuning:
    speed: float = VOICE_SPEED_DEFAULT
    expressiveness: float = VOICE_EXPRESSIVENESS_DEFAULT

    @classmethod
    def from_payload(cls, payload: dict | None) -> "VoiceTuning":
        """Build from whatever a client sent, keeping defaults for absent keys.

        Absent and unreadable are treated the same on purpose: a client that
        knows nothing about voice settings sends neither key and gets the
        default voice, which is exactly what a malformed value should also get.
        """
        data = payload or {}
        return cls(
            speed=_clamp(data.get("speed"), VOICE_SPEED_MIN, VOICE_SPEED_MAX, VOICE_SPEED_DEFAULT),
            expressiveness=_clamp(data.get("expressiveness"), 0.0, 1.0, VOICE_EXPRESSIVENESS_DEFAULT),
        )

    @property
    def is_default(self) -> bool:
        """Whether this asks for nothing the voice wouldn't do on its own.

        Worth knowing: leaving `voice_settings` off an ElevenLabs call keeps
        whatever is saved on the voice itself, which is not necessarily the API
        defaults. Sending nothing when nothing was asked for means the untouched
        agent still sounds exactly as it did before any of this existed.
        """
        return (
            abs(self.speed - VOICE_SPEED_DEFAULT) < 1e-6
            and abs(self.expressiveness - VOICE_EXPRESSIVENESS_DEFAULT) < 1e-6
        )

    @property
    def stability(self) -> float:
        """ElevenLabs' axis: 1.0 is flat and even, 0.0 is at its most animated."""
        return round(1.0 - self.expressiveness, 4)

    def as_dict(self) -> dict:
        return {"speed": self.speed, "expressiveness": self.expressiveness}


DEFAULT_TUNING = VoiceTuning()
