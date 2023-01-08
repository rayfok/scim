
import springs as sp

from .visualization import TypedSentsViz
from .make_output import write_sentences_to_json, write_all_to_json
from .pipeline import Pipeline, PipelineConfig


@sp.dataclass
class AppConfig:
    pipeline: PipelineConfig = PipelineConfig()
    src: str = sp.MISSING
    dst: str = sp.MISSING
    mode: str = sp.MISSING


@sp.cli(AppConfig)
def main(config: AppConfig):
    pipeline = Pipeline(config.pipeline)
    doc = pipeline(config.src)

    if config.mode == 'viz':
        TypedSentsViz()(doc=doc, path=config.dst)
    elif config.mode == 'all':
        write_all_to_json(doc=doc, dst=config.dst)
    elif config.mode == 'sent':
        write_sentences_to_json(doc=doc, dst=config.dst, src=config.src)
    else:
        raise ValueError(f"Invalid mode: {config.mode}")


if __name__ == '__main__':
    main()
