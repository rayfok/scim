import re
from typing import List

from mmda.predictors.heuristic_predictors.dictionary_word_predictor import \
    DictionaryWordPredictor
from mmda.types.annotation import BoxGroup, SpanGroup
from mmda.types.document import Document


class ExtendedDictionaryWordPredictor(DictionaryWordPredictor):

    def run_small_caps_heuristic(
        self: 'ExtendedDictionaryWordPredictor',
        lead: SpanGroup,
        tail: SpanGroup,
    ) -> bool:
        assert lead.text is not None and tail.text is not None, \
            "SpanGroup must have text."

        # check if both spans contain capital letters
        is_both_all_caps = (
            re.match(r'^[\-\(\[]{0,1}[A-Z\-]+$', lead.text)
            is not None and
            re.match(r'^[A-Z\-]+[\)\]]{0,1}[\-,\.\?\!\;\:]{0,1}$', tail.text)
            is not None
        )

        # if this is a word with small caps, then the last span for the first
        # group and the first span for the second group should have different
        # letter heights

        is_different_heights = (
            lead.spans[-1].box.h != tail.spans[0].box.h  # type: ignore
        )

        return is_both_all_caps and is_different_heights

    def predict(self: 'ExtendedDictionaryWordPredictor',
                document: Document) -> List[SpanGroup]:
        """Get words from a document as a list of SpanGroup.
        """

        words = super().predict(document)

        i = 0  # count explicitly because length of `words` is changing
        while i < (len(words) - 1):
            to_merge = self.run_small_caps_heuristic(lead := words[i],
                                                     tail := words[i + 1])

            if to_merge:
                # spans are simply concatenated
                new_spans = lead.spans + tail.spans

                # bit of logic to determine if any of the spans to merge have
                # attribute box_group set to not None, and if so, deal with
                # merging them properly.
                if lead_bg := lead.box_group:
                    if tail_bg := tail.box_group:
                        new_box_groups = BoxGroup(
                            boxes=(lead_bg.boxes + tail_bg.boxes),
                            type=(lead_bg.type or tail_bg.type)
                        )
                    else:
                        new_box_groups = lead.box_group
                elif tail.box_group:
                    new_box_groups = tail.box_group
                else:
                    new_box_groups = None

                # the new text for the merge span group is the concatenation
                # of the text of the two spans if at least one has text,
                # otherwise it is None
                new_text = ((lead.text or '') + (tail.text or '') or None)

                # we give lead token precedence over tail token in type
                new_type = (lead.type or tail.type)

                # make new span group, replace the first of the two existing
                # ones, then toss away the second one.
                merged = SpanGroup(spans=new_spans,
                                   id=i,
                                   text=new_text,
                                   type=new_type,
                                   box_group=new_box_groups)
                words[i] = merged
                words.pop(i + 1)

            else:
                i += 1
                # refresh the word id bc list will (potentially) get shorter
                # as we merge
                words[i].id = i

        return words
