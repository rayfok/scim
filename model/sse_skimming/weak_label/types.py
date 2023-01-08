from dataclasses import dataclass
from typing import Optional
from enum import IntEnum


@dataclass
class Instance:
    id: int
    type: str
    text: str
    section: str
    block_id: str
    span_start: int
    span_end: int
    doc_id: str = ""
    label: Optional[str] = None
    score: Optional[int] = None

class Label(IntEnum):
    ABSTAIN = -1
    NONSIG = 0
    METHOD = 1
    RESULT = 2
    NOVELTY = 3
