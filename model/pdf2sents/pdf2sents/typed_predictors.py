import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Dict

from mmda.predictors.base_predictors.base_predictor import BasePredictor
from mmda.types.annotation import BoxGroup, SpanGroup, Annotation, Span
from mmda.types.document import Document
from mmda.types.names import Words, Sentences, Blocks
import tqdm

from .layout_tools import (
    intersect_span_groups,
    span_is_fully_contained,
    box_group_from_span_group,
)
from .types import TypedBlocks


def make_typed_span_group(
    spans: List[Span],
    document: Document,
    type_: Optional[str] = None,
    id_: Optional[int] = None,
    add_text: bool = True,
) -> SpanGroup:
    typed_sg = SpanGroup(
        spans=spans,
        id=id_,
        type=type_ or 'Other',
    )
    typed_sg.box_group = box_group_from_span_group(
        span_group=typed_sg, doc=document
    )
    if add_text:
        typed_sg.text = ' '.join(
            str(word.text) for word in
            document.find_overlapping(typed_sg, Words)
        )
    return typed_sg


class TypedBlockPredictor(BasePredictor):
    REQUIRED_BACKENDS = None                        # type: ignore
    REQUIRED_DOCUMENT_FIELDS = [Blocks, Words]      # type: ignore

    Text = 'Text'
    Title = 'Title'
    ListType = 'List'
    Table = 'Table'
    Figure = 'Figure'
    Other = 'Other'
    RefApp = 'ReferencesAppendix'
    Abstract = 'Abstract'
    Preamble = 'Preamble'
    Caption = 'Caption'

    def _get_block_words(self,
                         doc: Document,
                         block: SpanGroup) -> Sequence[str]:
        words = doc.find_overlapping(block, Words)
        return [str(w.text if w.text else '') for w in words]

    def _tag_abstract_blocks(self, doc: Document, blocks: List[SpanGroup]):
        in_abstract = False
        abstract_position = -1
        for i, block in enumerate(blocks):
            skip_block = False
            if (
                block.box_group is not None
                and block.box_group.type == self.Title
            ):
                sec_type = re.sub(
                    r'^(\d|[\.])+\s+', '',
                    # remove leading section numbers if present
                    ' '.join(self._get_block_words(doc, block))
                ).lower()
                if abstract_position >= 0:
                    break

                # HEURISTIC only check for match in the first 20 chars or so
                if 'abstract' in sec_type[:20]:
                    abstract_position = i
                    in_abstract = True

                    # We skip the actual title of the abstract section,
                    # which should just be "Abstract"
                    skip_block = True

            if in_abstract and not skip_block:
                block.type = self.Abstract

        # mark everything before first abstract as preamble
        if abstract_position > 0:
            for block in blocks[:abstract_position]:
                block.type = self.Preamble
        elif (
            abstract_position == 0 and
            (abstract_start := blocks[0].start) > 0
        ):
            # make a preamble block if the first recognized block has
            # zero index but is not at the first position in the document
            preamble_sg = make_typed_span_group(
                spans=[Span(start=0, end=abstract_start)],
                document=doc,
                type_=self.Preamble,
                add_text=False,
            )
            blocks.insert(0, preamble_sg)

    def _tag_references_blocks(
        self,
        doc: Document,
        blocks: Sequence[SpanGroup]
    ) -> None:
        in_references = False
        for block in blocks:
            skip_block = False

            if (
                block.box_group is not None and (
                    block.box_group.type == self.Title
                    # HEURISTIC sometimes the title of the references section
                    # is incorrectly tagged as a list, so we check for that
                    # type as well.
                    or block.box_group.type == self.ListType
                )
            ):
                sec_type = re.sub(
                    # remove leading section numbers if present
                    r'^(\d|[\.])+\s+', '',
                    ' '.join(self._get_block_words(doc, block))
                ).lower()
                # HEURISTIC only check for match in the first 20 chars or so
                if 'references' in sec_type[:20]:
                    in_references = True

                    if block.box_group.type == self.Title:
                        # We skip the actual title of the references section,
                        # which should just be "References"
                        skip_block = True

            if in_references and not skip_block:
                block.type = self.RefApp

    def _tag_caption_blocks(self, doc: Document, blocks: List[SpanGroup]):
        for block in blocks:
            if (
                block.box_group is not None and
                # blocks that are tagged as either a title or a text
                # are very likely to be captions
                (block.box_group.type == self.Title
                 or block.box_group.type == self.Text)
            ):
                # HEURISTIC: only look at the first 20 chars or so of the block
                block_text = ' '.join(self._get_block_words(doc, block))[:20]

                # HEURISTIC: look for either 'Figure' or 'Table' in the block
                # text to determine if it is a caption
                is_table_or_figure_caption = (
                    re.match(r'^[Tt]able \d+[a-z]?\:', block_text)
                    or re.match(r'^[Ff]igure \d+[a-z]?\:', block_text)
                )

                # TODO: after MMDA adds bold text to properties, consider
                # checking for that too!

                if is_table_or_figure_caption:
                    block.type = self.Caption

    def _create_typed_blocks(self, doc: Document) -> List[SpanGroup]:
        cur_blocks: List[SpanGroup] = getattr(doc, 'blocks', [])
        new_blocks: List[SpanGroup] = []

        for block in tqdm.tqdm(
            cur_blocks,
            desc='Typing blocks',
            unit=' blocks',
            unit_scale=True
        ):
            block_type = None

            if len(block.spans) < 1 or block.box_group is None:
                continue
            elif (
                block.box_group.type == self.Text or
                block.box_group.type == self.ListType
            ):
                block_type = block.box_group.type
            elif block.box_group.type == self.Title:
                cur_sents: Iterable[SpanGroup] = block.sents   # type: ignore

                sents = [
                    sent for sent in cur_sents if
                    span_is_fully_contained(block, sent)
                ]

                if len(sents) >= 2:
                    # HEURISTIC: something tagged as a title with at
                    # least two fully contained sentences is probably a text
                    block_type = self.Text
                else:
                    block_type = self.Title
            else:
                block_type = block.box_group.type or self.Other

            new_blocks.append(
                SpanGroup(
                    spans=block.spans,
                    id=len(new_blocks),
                    type=str(block_type),
                    box_group=BoxGroup(
                        boxes=block.box_group.boxes,
                        type=block.box_group.type
                    )
                )
            )
        return new_blocks

    def predict(self, document: Document) -> Sequence[Annotation]:
        typed_blocks = self._create_typed_blocks(document)
        self._tag_abstract_blocks(document, typed_blocks)
        self._tag_references_blocks(document, typed_blocks)
        self._tag_caption_blocks(document, typed_blocks)
        return typed_blocks


class TypedSentencesPredictor(BasePredictor):
    REQUIRED_BACKENDS = None                                # type: ignore
    REQUIRED_DOCUMENT_FIELDS = [TypedBlocks, Sentences]     # type: ignore

    CONTENT_TYPES: Set[str] = {
        TypedBlockPredictor.Text,
        TypedBlockPredictor.ListType,
        TypedBlockPredictor.Abstract,
        TypedBlockPredictor.Caption,
        TypedBlockPredictor.RefApp,
    }
    LAYOUT_TYPES: Set[str] = {
        TypedBlockPredictor.Title,
        TypedBlockPredictor.Table,
        TypedBlockPredictor.Figure,
        TypedBlockPredictor.Preamble
    }

    def predict(self, document: Document) -> List[SpanGroup]:
        """Given a document with typed_blocks and sentence annotations,
        run a series of heuristics to assign type labels to individual
        sentences, as well as split certain sentences that span across
        multiple blocks.

        Args:
            document (Document): A document with typed_blocks and sentence
                annotations.
        """

        # Typing annotations so that mypy does not freak out
        typed_block: SpanGroup
        sent: SpanGroup

        # this is where we will accumulate the sentences with types
        typed_sents: List[SpanGroup] = []

        # This dictionary contains the index of the last position a sentence
        # was sliced on; the key is a tuple of (start_of_unsliced,
        # end_of_unsliced). If a sentence is not sliced, then the value
        # is the end position.
        typed_sents_ends: Dict[Tuple[int, int], int] = {}

        prog = tqdm.tqdm(
            desc='Typing sentences',
            unit=' sent',
            unit_scale=True
        )

        for typed_block in document.typed_blocks:   # type: ignore
            is_content_block = typed_block.type in self.CONTENT_TYPES

            for sent in typed_block.sents:          # type: ignore
                if typed_block.type in self.LAYOUT_TYPES:
                    # CASE 1: This sentence is part of a title, table,
                    #         figure, or preamble. Therefore, because it is
                    #         part of a visual block, we want to make
                    #         this sentence as tight as possible, meaning not
                    #         letting it span across other blocks.
                    tight_spans = intersect_span_groups(typed_block, sent)
                    tight_sg = make_typed_span_group(
                        spans=tight_spans,
                        document=document,
                        type_=typed_block.type,
                        id_=len(typed_sents)
                    )
                    add_to_typed_sentences = (
                        (
                            key := (sent.start, sent.end)
                        ) not in typed_sents_ends
                        or typed_sents_ends[key] < tight_sg.end
                    )

                    if add_to_typed_sentences:
                        prog.update(1)
                        typed_sents_ends[key] = tight_sg.end
                        typed_sents.append(tight_sg)
                    # # # # # # END OF CASE 1 # # # # # #

                elif sent.start < typed_block.start:
                    # CASE 2: This sentence starts before the current block.
                    #         We have a few more checks to do before being able
                    #         to add it to the typed sentences.

                    previous_blocks_are_all_visual = all(
                        b.type in self.LAYOUT_TYPES
                        for b in sent.typed_blocks
                        if b.start < typed_block.start  # type: ignore
                    )
                    already_sliced = (
                        (key := (sent.start, sent.end)) in typed_sents_ends
                    )

                    if is_content_block and (
                        previous_blocks_are_all_visual or already_sliced
                    ):
                        # CASE 2a: We add this sentence to the one extracted
                        #          from this block because (a) the current
                        #          block is a content block, meaning text,
                        #          list, or abstract, and (b) the sentence
                        #          belongs to all visual blocks except the
                        #          current block.
                        tight_spans = intersect_span_groups(typed_block, sent)
                        tight_sg = make_typed_span_group(
                            spans=tight_spans,
                            document=document,
                            type_=typed_block.type,
                            id_=len(typed_sents)
                        )
                        if typed_sents_ends.get(key, -1) < tight_sg.end:
                            prog.update(1)
                            typed_sents_ends[key] = tight_sg.end
                            typed_sents.append(tight_sg)

                        # # # # # # END OF CASE 2 # # # # # #

                elif (key := (sent.start, sent.end)) not in typed_sents_ends:
                    # CASE 3: This sentence starts in the current block.
                    #         We add it to the typed sentences unless it has
                    #         been added already.
                    new_sg = make_typed_span_group(
                        spans=sent.spans,
                        type_=typed_block.type,
                        document=document,
                        id_=len(typed_sents)
                    )
                    prog.update(1)
                    typed_sents_ends[key] = new_sg.end
                    typed_sents.append(new_sg)
                    # # # # # # END OF CASE 3 # # # # # #

        # CASE 4: We finish by adding all sentences that are not part of
        #         any block; these will be tagged with type 'Other'.
        for sent in document.sents:     # type: ignore
            if (sent_key := (sent.start, sent.end)) not in typed_sents_ends:
                prog.update(1)

                new_sg = make_typed_span_group(
                    spans=sent.spans,
                    type_=TypedBlockPredictor.Other,
                    document=document,
                    id_=len(typed_sents)
                )
                typed_sents_ends[sent_key] = new_sg.end
                typed_sents.append(new_sg)
        # # # # # # END OF CASE 4 # # # # # #

        return typed_sents
