from dataclasses import dataclass
from typing import List, Optional

from mmda.types.document import Document

from sse_skimming.heuristic_utils import (
    clean_sentence,
    make_sent_block_map,
    make_sent_sect_map,
)


@dataclass
class Instance:
    id: str
    type: str
    text: str
    section: str
    block_id: str
    span_start: int
    span_end: int
    label: Optional[str] = None
    score: Optional[int] = None


def build_dataset(doc: Document, valid_types: List[str]) -> List[Instance]:
    dataset = []
    sent_sect_map, _ = make_sent_sect_map(doc.typed_sents, valid_types)
    sent_block_map = make_sent_block_map(doc.typed_sents, doc.typed_blocks)
    for sg in doc.typed_sents:
        if sg.type in valid_types:
            dataset.append(
                Instance(
                    id=sg.id,
                    type=sg.type,
                    text=clean_sentence(sg.text),
                    section=sent_sect_map[sg.id],
                    block_id=sent_block_map[sg.id],
                    span_start=sg.spans[0].start,
                    span_end=sg.spans[0].end,
                )
            )
    return dataset
