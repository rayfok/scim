from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from PIL.Image import Image as PILImage

from mmda.types.annotation import SpanGroup, Span

from layoutparser.elements.layout_elements import (
    TextBlock,
    Rectangle
)
from layoutparser.visualization import draw_box

from mmda.types.document import Document
from mmda.types.names import Images, Pages

from .layout_tools import box_group_from_span_group


class BaseViz:
    def __init__(self, color_map: Dict[str, str], attribute: str):
        self.color_map = color_map
        self.attribute = attribute

    @classmethod
    def get_span_group_type(cls, span_group: SpanGroup) -> Union[str, None]:
        if span_group.type is not None:
            return span_group.type
        elif span_group.box_group is not None:
            return span_group.box_group.type
        else:
            return None

    @staticmethod
    def _yield_alpha(alpha: Union[float, List[float]]) -> Iterable[float]:
        if isinstance(alpha, float):
            while True:
                yield alpha
        else:
            yield from alpha

    @classmethod
    def draw_blocks(
        cls,
        image: PILImage,
        doc_spans: List[SpanGroup],
        pid: Optional[int] = None,
        color_map: Optional[Dict[str, str]] = None,
        token_boundary_width: int = 0,
        alpha: Union[float, List[float]] = 0.3,
        **kwargs,
    ):

        w, h = image.size
        layout = []
        alpha_values = []

        for span, alpha_val in zip(doc_spans, cls._yield_alpha(alpha)):
            for box in getattr(span.box_group, 'boxes', []):
                if box.page != pid or pid is None:
                    continue
                text_block = TextBlock(
                    Rectangle(
                        *box
                        .get_absolute(page_height=h, page_width=w)
                        .coordinates
                    ),
                    type=cls.get_span_group_type(span),
                    text=(doc_spans[0].text or ' '.join(span.symbols)),
                )
                layout.append(text_block)
                alpha_values.append(alpha_val)

        return draw_box(
            image,
            layout,      # type: ignore
            color_map=color_map,
            box_color='grey' if not color_map else None,
            box_width=token_boundary_width,
            box_alpha=alpha_values,
            **kwargs,
        )

    def _check_attrs(self, doc: Document):
        if not(hasattr(doc, Pages) and hasattr(doc, Images)):
            raise ValueError(
                f'Document must have `{Pages}` and`{Images}` annotations!'
            )

    def __call__(self, doc: Document, path: Union[str, Path]):
        path = Path(path)

        self._check_attrs(doc)

        pages: List[SpanGroup] = getattr(doc, 'pages', [])
        images: List[PILImage] = getattr(doc, 'images', [])

        for pid in range(len(pages)):
            viz = self.draw_blocks(
                image=images[pid],
                doc_spans=getattr(pages[pid], self.attribute, []),
                pid=pid,
                color_map=self.color_map
            )

            path.with_suffix("").mkdir(parents=True, exist_ok=True)
            viz.save(path.with_suffix("") / f"{pid}.png")


class TypedSentsViz(BaseViz):
    def __init__(self):
        from .typed_predictors import TypedBlockPredictor
        color_map = {
            TypedBlockPredictor.Title: 'red',
            TypedBlockPredictor.Text: 'blue',
            TypedBlockPredictor.Figure: 'green',
            TypedBlockPredictor.Table: 'yellow',
            TypedBlockPredictor.ListType: 'orange',
            TypedBlockPredictor.Other: 'grey',
            TypedBlockPredictor.RefApp: 'purple',
            TypedBlockPredictor.Abstract: 'magenta',
            TypedBlockPredictor.Preamble: 'cyan',
            TypedBlockPredictor.Caption: 'pink'
        }

        super().__init__(color_map=color_map, attribute='typed_sents')


class VizAny(BaseViz):
    def __init__(self, color_map: Dict[str, str]):
        super().__init__(color_map=color_map, attribute='')

    def __call__(
        self,
        doc: Document,
        path: Union[str, Path],
        spans: List[SpanGroup],
        opacity: Optional[List[float]] = None,
    ):
        path = Path(path)

        assert opacity is None or len(opacity) == len(spans), \
            'opacity must be None or have the same length as spans'

        self._check_attrs(doc)
        pages: List[SpanGroup] = getattr(doc, 'pages', [])
        images: List[PILImage] = getattr(doc, 'images', [])

        for pid in range(len(pages)):

            # collect here only annotations for this page
            page_spans, page_opacity = [], []
            for i, sg in enumerate(spans):
                if (
                    # either this sentence starts on this page...
                    pages[pid].start <= sg.start <= pages[pid].end
                    # or it ends on this page!
                    or pages[pid].start <= sg.end <= pages[pid].end
                ):

                    sg_spans = [Span(start=max(sg.start, pages[pid].start),
                                     end=min(sg.end, pages[pid].end))]
                    new_sg = SpanGroup(
                        spans=sg_spans,
                        type=sg.type,
                        text=sg.text
                    )
                    new_sg.box_group = box_group_from_span_group(sg)
                    page_spans.append(new_sg)
                    page_opacity.append(opacity[i] if opacity else 0.3)

            viz = self.draw_blocks(
                image=images[pid],
                doc_spans=page_spans,
                pid=pid,
                color_map=self.color_map,
                alpha=page_opacity
            )

            path.with_suffix("").mkdir(parents=True, exist_ok=True)
            viz.save(path.with_suffix("") / f"{pid}.png")
