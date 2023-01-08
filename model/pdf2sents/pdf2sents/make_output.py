import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from uuid import uuid4


from mmda.types.document import Document
from mmda.types.annotation import SpanGroup


def get_sha_of_pdf(path: Path) -> str:
    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    sha1 = hashlib.sha1()

    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()


@dataclass
class PaperSentences:
    docid: str
    texts: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    senid: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, content: Dict[str, Any]) -> 'PaperSentences':
        return cls(**content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'docid': self.docid,
            'texts': self.texts,
            'types': self.types,
            'senid': self.senid,
        }

    def add_sentence(self, text: str, type: str, senid: int):
        self.texts.append(text)
        self.types.append(type)
        self.senid.append(senid)

    @classmethod
    def from_document(
        cls,
        doc: Document,
        docid: Optional[str] = None
    ) -> 'PaperSentences':

        docid = docid or uuid4().hex
        content = cls(docid=docid)

        for sent in doc.typed_sents:    # type: ignore
            content.add_sentence(
                text=sent.text, type=sent.type, senid=sent.id   # type: ignore
            )
        return content


def write_sentences_to_json(
    src: Union[Path, str],
    doc: Document,
    dst: Union[Path, str]
):
    sent: SpanGroup

    dst = Path(dst).with_suffix(".jsonl")
    sha = get_sha_of_pdf(Path(src))

    data = []
    if dst.exists():
        with open(dst, 'r') as f:
            for ln in f:
                paper_sentences = PaperSentences.from_dict(json.loads(ln))
                if paper_sentences.docid != sha:
                    data.append(paper_sentences)

    data.append(PaperSentences.from_document(doc=doc, docid=sha))

    with open(dst, 'w') as f:
        for paper_sentences in data:
            content = json.dumps(paper_sentences.to_dict(), sort_keys=True)
            f.write(f'{content}\n')


def write_all_to_json(doc: Document, dst: Union[Path, str]):
    dst = Path(dst).with_suffix(".json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(doc.to_json(), f, sort_keys=True)
