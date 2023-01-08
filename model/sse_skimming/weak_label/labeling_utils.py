import json
import os
from glob import glob
from pathlib import Path
from typing import List
from collections import defaultdict

from weak_label.types import Instance


def load_dataset(src: str) -> List[Instance]:
    dataset = []
    for filename in glob(os.path.join(src, "*.json")):
        doc_id = os.path.splitext(os.path.basename(filename))[0]
        with open(filename, "r") as f:
            for x in json.load(f):
                dataset.append(Instance(**x, doc_id=doc_id))
    return dataset


def export_labeled_dataset(
    dataset: List[Instance],
    dst: str,
    text_and_labels_only: bool = True,
    filter_abstain: bool = False,
) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if filter_abstain:
        dataset = [x for x in dataset if x.label != -1]
    if text_and_labels_only:
        by_doc = defaultdict(list)
        for x in dataset:
            by_doc[x.doc_id].append(x)
        with open(dst.with_suffix(".jsonl"), "w") as out:
            for instances in by_doc.values():
                out.write(
                    json.dumps(
                        {
                            "sents": [x.text for x in instances],
                            "labels": [x.label for x in instances],
                        }
                    ) + "\n"
                )
    else:
        with open(dst.with_suffix(".json"), "w") as out:
            json.dump(dataset, out, default=lambda x: x.__dict__)
