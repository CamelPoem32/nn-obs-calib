"""Single-window preprocessing and neural observability prediction."""

from dataclasses import dataclass

from obscalib.data.structures import TokenBatch, WindowBatch
from obscalib.geometry.processing import GeometryProcessor
from obscalib.models.observability_prediction_model import ObservabilityPredictionModel
from obscalib.models.structures import ObservabilityPredictionOutput
from obscalib.tokenization.tokenizer import Tokenizer


@dataclass
class ObservabilityPredictionStepResult:
    """Outputs of one observability-prediction window."""

    output: ObservabilityPredictionOutput
    tokens: TokenBatch


class ObservabilityPredictionStep:
    """Run geometry preprocessing, calibration-aware tokenization, and observability prediction."""

    def __init__(self, geometry_processor: GeometryProcessor, tokenizer: Tokenizer, model: ObservabilityPredictionModel) -> None:
        self.geometry_processor = geometry_processor
        self.tokenizer = tokenizer
        self.model = model

    def __call__(self, window: WindowBatch) -> ObservabilityPredictionStepResult:
        canonical_streams = self.geometry_processor(window.streams, window.metadata, window.current_calibration)
        tokens = self.tokenizer(canonical_streams, observability=None)
        tokens.validate()
        output = self.model(tokens)

        return ObservabilityPredictionStepResult(output=output, tokens=tokens)