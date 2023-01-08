import re

import spacy
from pdf2sents.typed_predictors import TypedBlockPredictor
from spacy.symbols import VERB

from sse_skimming.word_bank import *

nlp = spacy.load("en_core_web_sm")


def sentence_has_author_intent(sent):
    return re.search(r"(?=\b(" + "|".join(AUTHOR_KWS) + r")\b)", sent.text.lower())


def sent_contains_novelty(sent):
    for token in nlp(sent.text):
        if token.lemma_ in NOVELTY_KWS and is_sentence_in_section(
            sent,
            ["abstract", "introduction", "related work", "recent work", "conclusion"],
        ):
            return True
    return False


def sent_contains_objective(sent):
    for token in nlp(sent.text):
        if token.lemma_ in OBJECTIVE_KWS and is_sentence_in_section(
            sent,
            ["abstract", "introduction", "related work", "recent work", "conclusion"],
        ):
            return True
    return False


def sent_contains_contribution(sent):
    for token in nlp(sent.text):
        if token.lemma_ in CONTRIBUTION_KWS and is_sentence_in_section(
            sent, ["abstract", "introduction", "conclusion"]
        ):
            return True
    return False


def sent_contains_result(sent):
    for token in nlp(sent.text):
        if token.lemma_ in RESULT_KWS:
            return True
    return False


def classify_novelty_batch(ref_sents):
    return {s.id for s in ref_sents if sent_contains_novelty(s)}


def classify_objective_batch(ref_sents):
    return {s.id for s in ref_sents if sent_contains_objective(s)}


def classify_method_batch(ref_sents):
    pass


def classify_result_batch(ref_sents):
    return {s.id for s in ref_sents if sent_contains_result(s)}


def is_sentence_in_section(sentence, sections):
    return any(section in sentence.section.lower() for section in sections)


def make_sent_sect_map(sents, valid_types):
    sent_by_id = {s.id: s for s in sents}
    sents = sorted(sents, key=lambda x: x.spans[0].start)
    sent_sect_map = {}
    sent_span_map = {}  # map sentence to starting span location
    sect_box_map = {}  # map section to first bbox location

    sections = {}  # map section number to section header text
    current_section = "Abstract"
    for sent in sents:
        if sent.type in valid_types:
            sent_sect_map[sent.id] = current_section
        elif sent.type == TypedBlockPredictor.Title:
            current_section = sent.text
            match = re.search(r"[-+]?\d*\.\d+|\d+", current_section)
            if match:
                sect_number = match.group()
                sections[sect_number] = current_section
            sect_box_map[current_section] = sent.box_group.boxes[0]
        sent_span_map[sent.id] = sent.spans[0].start

    for sent_id, sect in sent_sect_map.items():
        match = re.search(r"[-+]?\d*\.\d+|\d+", sect)
        if match:
            sect_number = match.group()
            sect_number_parts = sect_number.split(".")

            if len(sect_number_parts) == 3:
                first_level_header = sections.get(sect_number_parts[0], "")
                second_level_header = sections.get(".".join(sect_number_parts[:2]), "")
                sent_sect_map[
                    sent_id
                ] = f"{first_level_header} @@ {second_level_header} @@ {sect}"
            elif len(sect_number_parts) == 2:
                first_level_header = sections.get(sect_number_parts[0], "")
                sent_sect_map[sent_id] = f"{first_level_header} @@ {sect}"

    # If there's no introduction section detected, it was probably misclassified
    # as part of the abstract. We try to disentangle the two sections here.
    has_intro = False
    for section in sections.keys():
        if "introduction" in section.lower():
            has_intro = True
            break
    if not has_intro:
        # Try to find "Introduction"
        intro_sent = None
        for sent_id, section in sent_sect_map.items():
            sent = sent_by_id[sent_id]
            if "abstract" not in section.lower():
                continue
            found_intro = re.search(r"(?i)\d+\s?Introduction", sent.text)
            if found_intro:
                intro_sent = sent
                break

        if intro_sent:
            intro_span_start = intro_sent.spans[0].start
            for sent_id, section in sent_sect_map.items():
                if "abstract" in section.lower():
                    if sent_by_id[sent_id].spans[0].start >= intro_span_start:
                        sent_sect_map[sent_id] = "1 Introduction"
            sect_box_map["1 Introduction"] = intro_sent.box_group.boxes[0]

    return sent_sect_map, sect_box_map


def make_sent_block_map(sents, typed_blocks):
    blocks = [(b.spans[0].start, b.spans[0].end, b.uuid) for b in typed_blocks]
    blocks = sorted(blocks, key=lambda b: b[1])

    sent_block_map = {}
    for sent in sents:
        for block_span_range in blocks:
            _, block_span_end, block_uuid = block_span_range
            if sent.spans[0].start <= block_span_end:
                sent_block_map[sent.id] = block_uuid
                break

    return sent_block_map


def clean_sentence(sentence):
    cleaned = sentence.replace("\ufb01", "fi")
    cleaned = cleaned.replace("\ufb02", "fl")
    cleaned = cleaned.replace("\u2019", "'")
    cleaned = cleaned.replace("\u00ad", "-")
    cleaned = cleaned.encode("ascii", "ignore")
    cleaned = cleaned.decode()
    cleaned = cleaned.replace("- ", "")
    cleaned = cleaned.strip()
    return cleaned


def extract_verbs(sentence):
    verbs = []
    doc = nlp(sentence.text)
    for token in doc:
        if token.pos == VERB:
            verbs.append(token.lemma_)
    return verbs
