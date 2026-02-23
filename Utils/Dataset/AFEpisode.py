from dataclasses import dataclass
from typing import Optional

@dataclass
class AFEpisode:
    """Represents an AF episode with context"""
    episode_number: int
    start_sample: int
    end_sample: int
    start_time: float  # seconds
    end_time: float  # seconds
    duration: float  # seconds
    duration_minutes: float
    pre_rhythm: Optional[str]  # What rhythm was before AF
    available_sr_before: float  # Minutes of clean SR before onset
    starts_recording: bool  # Does this episode start the recording?
    ends_recording: bool  # Does this episode end the recording?