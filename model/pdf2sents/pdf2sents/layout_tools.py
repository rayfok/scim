import bisect
from itertools import chain
from typing import NamedTuple, Optional, Sequence, List, Dict, Union

from mmda.types.annotation import BoxGroup, SpanGroup
from mmda.types.box import Box
from mmda.types.span import Span
from mmda.types.document import Document
from mmda.types.names import Tokens


class SpanGroupSegments(NamedTuple):
    src: List[Span]
    dst: List[Span]
    both: List[Span]
    neither: List[Span]


def make_span_groups_segments(
    src: Union[SpanGroup, Sequence[Span]],
    dst: Union[SpanGroup, Sequence[Span]]
) -> SpanGroupSegments:
    """Horrible code to create span segments from two span groups.
    It slices two sequence of spans so that we create chunks that
    are either part of just the src or dst, both, or neither."""

    # if span groups are provided, get spans from span groups
    if isinstance(src, SpanGroup):
        src = src.spans
    if isinstance(dst, SpanGroup):
        dst = dst.spans

    # these are the locations at which each new segment starts or ends.
    segments_boundaries = sorted(set(chain.from_iterable(
        (s.start, s.end)
        for sg in (src, dst)
        for s in sg
    )))
    # this will be 0 if no span intersect this segment, 1 if only the src, 2 if
    # only the dst, and 3 if both.
    SRC_VAL = 1
    DST_VAL = 2
    segments_tags = [0] * len(segments_boundaries)

    for src_span in src:
        src_start = bisect.bisect_left(segments_boundaries, src_span.start)
        src_end = bisect.bisect_right(segments_boundaries, src_span.end)
        for i in range(src_start, src_end):
            segments_tags[i] += SRC_VAL

    for dst_span in dst:
        dst_start = bisect.bisect_left(segments_boundaries, dst_span.start)
        dst_end = bisect.bisect_right(segments_boundaries, dst_span.end)
        for i in range(dst_start, dst_end):
            segments_tags[i] += DST_VAL

    segments = SpanGroupSegments([], [], [], [])

    # let's iterate over all segments.
    i = 0
    while i < len(segments_tags):

        # this little loop here merges adjacent segments with the same tag.
        j = i
        while j < len(segments_tags):
            if segments_tags[j] != segments_tags[i]:
                break
            j += 1

        # now that we have found until when to merge, we can create the span.
        if j < len(segments_tags):
            # if we are not at the end of a sequence, we use start and end
            # positions to make the span.
            start, end = segments_boundaries[i], segments_boundaries[j]
        else:
            # if this is the last span segment in the sequence, we need
            # to backtrack to the previous segment to get the start position
            start, end = segments_boundaries[i - 1], segments_boundaries[-1]

        # actually making the span.
        new_span = Span(start, end)

        # we use the tags to figure out which type of span this is.
        if segments_tags[i] == SRC_VAL:
            segments.src.append(new_span)
        elif segments_tags[i] == DST_VAL:
            segments.dst.append(new_span)
        elif segments_tags[i] == SRC_VAL + DST_VAL:
            segments.both.append(new_span)
        else:
            segments.neither.append(new_span)

        # we restart from the same index where we found the end position
        # because the end in a span is exclusive.
        i = j

    return segments


def span_is_fully_contained(container: SpanGroup,
                            maybe_contained: SpanGroup) -> bool:
    return all(
        any(container_span.start <= maybe_contained_span.start
            and container_span.end >= maybe_contained_span.end
            for container_span in container.spans)
        for maybe_contained_span in maybe_contained.spans
    )


def intersect_span_groups(
    src: SpanGroup,
    dst: SpanGroup
) -> List[Span]:
    intersected_spans = []
    for src_span in src:
        for dst_span in dst:
            # check if there is an overlap between the two spans
            if (
                src_span.start <= dst_span.start <= src_span.end or
                src_span.start <= dst_span.end <= src_span.end or
                dst_span.start <= src_span.start <= dst_span.end or
                dst_span.start <= src_span.end <= dst_span.end
            ):
                intersected_spans.append(
                    Span(max(src_span.start, dst_span.start),
                         min(src_span.end, dst_span.end))
                )
    return intersected_spans


def difference_span_groups(
    src: SpanGroup,
    dst: SpanGroup
) -> List[Span]:
    diff_spans = []
    for src_span in src:
        for dst_span in dst:

            # check if there is an overlap between the two spans
            if (
                src_span.start <= dst_span.start <= src_span.end or
                src_span.start <= dst_span.end <= src_span.end or
                dst_span.start <= src_span.start <= dst_span.end or
                dst_span.start <= src_span.end <= dst_span.end
            ):
                diff_spans.append(
                    Span(max(src_span.start, dst_span.start),
                         min(src_span.end, dst_span.end))
                )
    return diff_spans


class BoxKey(NamedTuple):
    top: float
    page: int


def box_group_from_span_group(
    span_group: SpanGroup,
    doc: Optional[Document] = None,
    merge_boxes: bool = True,
    digits: int = 2
) -> BoxGroup:

    doc = span_group.doc or doc
    if doc is None:
        raise ValueError('span_group must have a doc, or doc must be provided')

    boxes_it: Sequence[Box] = (
        span.box    # type: ignore
        for token in doc.find_overlapping(span_group, Tokens)
        for span in token.spans
    )
    to_merge: Dict[BoxKey, List[Box]] = {}

    for box in boxes_it:
        to_merge.setdefault(
            BoxKey(top=round(box.t, digits), page=box.page),
            []
        ).append(box)

    merged_boxes = []
    for box_key, boxes in to_merge.items():
        merged_boxes.append(
            Box(
                l=(left := min(box.l for box in boxes)),
                t=(top := min(box.t for box in boxes)),
                w=(max(box.l + box.w for box in boxes) - left),
                h=(max(box.t + box.h for box in boxes) - top),
                page=box_key.page,
            )
        )
    return BoxGroup(boxes=merged_boxes, type=span_group.type)
