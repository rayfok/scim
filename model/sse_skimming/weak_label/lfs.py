import re
from typing import List

import numpy as np
import spacy
from pdf2sents.typed_predictors import TypedBlockPredictor
from snorkel.labeling import (LabelingFunction, LFAnalysis, LFApplier,
                              labeling_function)
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from snorkel.preprocess import preprocessor
from snorkel.preprocess.nlp import SpacyPreprocessor
from spaczz.matcher import FuzzyMatcher

from weak_label.types import Instance, Label
from weak_label.word_bank import WordBank


class Annotator:
    nlp = spacy.blank("en")
    spacy_preprocessor = SpacyPreprocessor(
        text_field="text", doc_field="doc", memoize=True, memoize_key=id
    )
    wordbank = WordBank()

    def _keyword_lookup(x, keywords, label):
        if any(word in x.text.lower() for word in keywords):
            return label
        return Label.ABSTAIN

    def _make_keyword_lf(keywords, label):
        return LabelingFunction(
            name=f"keyword_{keywords[0]}",
            f=Annotator._keyword_lookup,
            resources=dict(keywords=keywords, label=label),
        )

    def _find_section_matches(x: Instance, sections: List[str]):
        matcher = FuzzyMatcher(Annotator.nlp.vocab)
        for section in sections:
            matcher.add(section, [Annotator.nlp(section)])
        matches = matcher(Annotator.nlp(x.section))
        return matches

    @preprocessor(memoize=True, memoize_key=lambda x: (x.doc_id, x.id))
    def has_author_statement(x: Instance) -> Instance:
        pattern = r"(?=\b(" + "|".join(Annotator.wordbank.AUTHOR_KWS) + r")\b)"
        has_author_statement = re.search(pattern, x.text.lower())
        x.has_author_statement = True if has_author_statement else False
        return x

    @preprocessor(memoize=True, memoize_key=lambda x: (x.doc_id, x.id))
    def is_in_acks(x: Instance) -> Instance:
        matches = Annotator._find_section_matches(x, ["acknowledgements"])
        x.is_in_acks = True if matches else False
        return x

    @preprocessor(memoize=True, memoize_key=lambda x: (x.doc_id, x.id))
    def has_link(x: Instance) -> Instance:
        x.has_link = any(word in x.text.lower() for word in ["http"])
        return x

    @preprocessor(memoize=True, memoize_key=lambda x: (x.doc_id, x.id))
    def has_arxiv(x: Instance) -> Instance:
        x.has_arxiv = any(word in x.text.lower() for word in ["arxiv"])
        return x

    @labeling_function(pre=[spacy_preprocessor])
    def exceeds_min_length(x: Instance) -> int:
        return Label.NONSIG if len(x.doc) <= 5 else Label.ABSTAIN

    @labeling_function(pre=[spacy_preprocessor])
    def exceeds_max_length(x: Instance) -> int:
        return Label.NONSIG if len(x.doc) >= 80 else Label.ABSTAIN

    @labeling_function(
        pre=[is_in_acks, has_link, has_arxiv, has_author_statement, spacy_preprocessor]
    )
    def contains_method_statement(x: Instance) -> int:
        ignore_sections = ["result", "experiment"]
        matches = Annotator._find_section_matches(x, ignore_sections)
        if matches:
            return Label.ABSTAIN

        if not x.has_author_statement or x.is_in_acks or x.has_link or x.has_arxiv:
            return Label.ABSTAIN
        for token in x.doc:
            if token.lemma_ in (
                Annotator.wordbank.RESULT_KWS
                | Annotator.wordbank.NOVELTY_KWS
                | Annotator.wordbank.OBJECTIVE_KWS
                | Annotator.wordbank.CONTRIBUTION_KWS
            ):
                return Label.ABSTAIN

        for token in x.doc:
            if token.pos_ == "VERB" and token.lemma_ in Annotator.wordbank.METHOD_KWS:
                return Label.METHOD
        return Label.ABSTAIN

    @labeling_function(pre=[is_in_acks, has_link, has_arxiv, has_author_statement])
    def is_in_method_section(x: Instance) -> int:
        if not x.has_author_statement or x.is_in_acks or x.has_link or x.has_arxiv:
            return Label.ABSTAIN
        sections = ["method", "setup", "approach"]
        matches = Annotator._find_section_matches(x, sections)
        return Label.METHOD if matches else Label.ABSTAIN

    @labeling_function(
        pre=[is_in_acks, has_link, has_arxiv, has_author_statement, spacy_preprocessor]
    )
    def contains_result_statement(x: Instance) -> int:
        if not x.has_author_statement or x.is_in_acks or x.has_link or x.has_arxiv:
            return Label.ABSTAIN
        sections = [
            "abstract",
            "introduction",
            "experiment",
            "result",
            "discussion",
            "conclusion",
        ]
        matches = Annotator._find_section_matches(x, sections)
        if not matches:
            return Label.ABSTAIN

        matches = Annotator._find_section_matches(x, ["setup"])
        if matches:
            return Label.ABSTAIN

        for token in x.doc:
            if token.lemma_ in (
                Annotator.wordbank.NOVELTY_KWS
                | Annotator.wordbank.OBJECTIVE_KWS
                | Annotator.wordbank.CONTRIBUTION_KWS
            ):
                return Label.ABSTAIN

        for token in x.doc:
            if token.lemma_ in Annotator.wordbank.RESULT_KWS:
                return Label.RESULT
        return Label.ABSTAIN

    @labeling_function(
        pre=[is_in_acks, has_link, has_arxiv, has_author_statement, spacy_preprocessor]
    )
    def contains_novelty_statement(x: Instance) -> int:
        if not x.has_author_statement or x.is_in_acks or x.has_link or x.has_arxiv:
            return Label.ABSTAIN
        sections = [
            "abstract",
            "introduction",
            "background",
            "related work",
            "recent work",
            "conclusion",
        ]
        matches = Annotator._find_section_matches(x, sections)
        if not matches:
            return Label.ABSTAIN

        # Objective statements should not include external reference
        if any(word in x.text.lower() for word in ["et al"]):
            return Label.ABSTAIN
        citation_pattern = r"\(19[0-9]\d|20[0-9]\d|2100\)"
        matches = re.search(citation_pattern, x.text.lower())
        if matches:
            return Label.ABSTAIN

        for token in x.doc:
            if (
                token.lemma_
                in Annotator.wordbank.NOVELTY_KWS
                | Annotator.wordbank.OBJECTIVE_KWS
                | Annotator.wordbank.CONTRIBUTION_KWS
            ):
                return Label.NOVELTY
        return Label.ABSTAIN

    @labeling_function()
    def is_in_acknowledgements(x: Instance) -> int:
        sections = ["acknowledgements"]
        matches = Annotator._find_section_matches(x, sections)
        return Label.NONSIG if matches else Label.ABSTAIN

    @labeling_function()
    def contains_link(x: Instance):
        lf = Annotator._make_keyword_lf(keywords=["http"], label=Label.NONSIG)
        return lf(x)

    @labeling_function()
    def contains_arxiv(x: Instance):
        lf = Annotator._make_keyword_lf(keywords=["arxiv"], label=Label.NONSIG)
        return lf(x)

    @labeling_function(pre=[has_author_statement])
    def is_list_elem_in_introduction(x: Instance) -> int:
        if x.type != TypedBlockPredictor.ListType:
            return Label.ABSTAIN
        matches = Annotator._find_section_matches(x, ["introduction"])
        return (
            Label.NOVELTY if matches and not x.has_author_statement else Label.ABSTAIN
        )

    @labeling_function(pre=[has_author_statement])
    def is_nonauthor_background(x: Instance) -> int:
        sections = ["background", "related work", "recent work"]
        matches = Annotator._find_section_matches(x, sections)
        return Label.NONSIG if matches and not x.has_author_statement else Label.ABSTAIN

    def apply_labeling_functions(self, dataset, lfs: List[LabelingFunction]):
        applier = LFApplier(lfs=lfs)
        return applier.apply(dataset)

    def get_lf_summary(self, labels: np.ndarray, lfs: List[LabelingFunction]):
        return LFAnalysis(L=labels, lfs=lfs).lf_summary()

    def predict(self, labels: np.ndarray, agg_model: str = "majority"):
        if agg_model == "majority":
            model = MajorityLabelVoter()
            return model.predict(L=labels)
        elif agg_model == "label":
            model = LabelModel(cardinality=4, verbose=True)
            model.fit(L_train=labels, n_epochs=500, log_freq=100, seed=69)
            return model.predict(labels, tie_break_policy="abstain")
