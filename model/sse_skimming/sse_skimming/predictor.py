from dataclasses import dataclass, field
from typing import Dict, List

import torch
from smashed.base.pipeline import Pipeline
from smashed.mappers.batchers import FixedBatchSizeMapper
from smashed.mappers.collators import FromTokenizerCollatorMapper
from smashed.mappers.converters import Python2TorchMapper
from smashed.mappers.fields import MakeFieldMapper
from smashed.mappers.multiseq import (
    AttentionMaskSequencePaddingMapper,
    MultiSequenceStriderMapper,
    SequencesConcatenateMapper,
    SingleValueToSequenceMapper,
    TokensSequencesPaddingMapper,
)
from smashed.mappers.tokenize import TokenizerMapper
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForTokenClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class PredictorConfig:
    sequence_max_length: int = 512
    tokenizer_input_field: str = "text"
    sentence_max_length: int = 80
    sentence_max_count: int = 10
    model_batch_size: int = 16
    model_ignore_labels_id: int = -100
    model_labels_names: List[str] = field(
        default_factory=lambda: ["background", "method", "objective", "other", "result"]
    )


Instance = Dict[str, List[str]]
Prediction = List[Dict[str, float]]


class Predictor:
    """
    Interface on to your underlying model.

    This class is instantiated at application startup as a singleton.
    You should initialize your model inside of it, and implement
    prediction methods.

    If you specified an artifacts.tar.gz for your model, it will
    have been extracted to `artifacts_dir`, provided as a constructor
    arg below.
    """

    config: PredictorConfig
    artifacts_dir: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    parser: Pipeline
    # _parser: SignificantSectionExtractionParser
    # _collator: DataCollatorForTokenClassification

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self.config = config
        self.artifacts_dir = artifacts_dir
        self._load_model()

    def _load_model(self) -> None:
        """
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.artifacts_dir,
            model_max_length=self.config.sequence_max_length,
        )

        self.parser = (
            TokenizerMapper(
                input_field=self.config.tokenizer_input_field,
                tokenizer=self.tokenizer,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.sentence_max_length,
            )
            >> MultiSequenceStriderMapper(
                max_stride_count=self.config.sentence_max_count,
                max_length=self.config.sequence_max_length,
                tokenizer=self.tokenizer,
                length_reference_field="input_ids",
            )
            >> TokensSequencesPaddingMapper(
                tokenizer=self.tokenizer, input_field="input_ids"
            )
            >> AttentionMaskSequencePaddingMapper(
                tokenizer=self.tokenizer, input_field="attention_mask"
            )
            >> MakeFieldMapper(field_name="labels", value=0, shape_like="input_ids")
            >> SingleValueToSequenceMapper(
                single_value_field="labels",
                like_field="input_ids",
                strategy="last",
                padding_id=self.config.model_ignore_labels_id,
            )
            >> SequencesConcatenateMapper()
            >> Python2TorchMapper()
            >> FixedBatchSizeMapper(batch_size=self.config.model_batch_size)
            >> FromTokenizerCollatorMapper(
                tokenizer=self.tokenizer,
                fields_pad_ids={"labels": self.config.model_ignore_labels_id},
            )
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.artifacts_dir,
            num_labels=len(self.config.model_labels_names),
        ).eval()

    def predict_one(self, instance: Instance) -> Prediction:
        """
        Should produce a single Prediction for the provided Instance.
        Leverage your underlying model to perform this inference.
        """
        samples = self.parser.map([instance])
        predictions: List[Dict[str, float]] = []

        # prediction = Prediction()
        with torch.no_grad():
            for batch in samples:
                output = self.model(**batch)

                # text = [self.tokenizer.decode(r) for r in batch['input_ids']]

                locs = batch["labels"] != self.config.model_ignore_labels_id
                scores = output.logits[locs]
                probs = torch.softmax(scores, dim=1)

                predictions.extend(
                    dict(zip(self.config.model_labels_names, sample_probs))
                    for sample_probs in probs.tolist()
                )

        return predictions

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.

        If your model gets performance benefits from batching during inference,
        implement that here, explicitly.

        Otherwise, you can leave this method as-is and just implement
        `predict_one()` above. The default implementation here passes
        each Instance into `predict_one()`, one at a time.

        The size of the batches passed into this method is configurable
        via environment variable by the calling application.
        """
        return [self.predict_one(instance) for instance in instances]
