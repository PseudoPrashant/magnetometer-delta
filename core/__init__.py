"""Core signal-processing building blocks (filters, rotation pipelines, axis signatures)."""

from core.axis_signature import AxisSignatureRecognizer, AxisSwipeEvent, ClusterMetrics
from core.filters import (
    ExponentialMovingAverage,
    LowPassFilter,
    OneEuroFilter,
    VectorStreamDenoiseFilter,
)
from core.kinematic_swipe import KinematicSwipeEvent, StreamKinematicSwipeDetector
from core.pipeline import (
    RotationPipeline,
    RotationPipelineV4,
    RotationPipelineV4_1,
    RotationPipelineV5,
    StrokeGestureRecognizer,
    ballistic_gain,
    ballistic_gain_smooth,
)

__all__ = [
    "AxisSignatureRecognizer",
    "AxisSwipeEvent",
    "ClusterMetrics",
    "ExponentialMovingAverage",
    "KinematicSwipeEvent",
    "LowPassFilter",
    "OneEuroFilter",
    "RotationPipeline",
    "RotationPipelineV4",
    "RotationPipelineV4_1",
    "RotationPipelineV5",
    "StreamKinematicSwipeDetector",
    "StrokeGestureRecognizer",
    "VectorStreamDenoiseFilter",
    "ballistic_gain",
    "ballistic_gain_smooth",
]
