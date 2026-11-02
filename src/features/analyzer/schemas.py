from pydantic import BaseModel, Field
from typing import Dict, List

class Keypoint(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    confidence: float

class FrameAnalysis(BaseModel):
    time: int = Field(..., description="Таймстемп кадра в миллисекундах")
    points: Dict[str, Keypoint] = Field(..., description="Имя точки -> Координаты")

class AnalysisResponse(BaseModel):
    frames: List[FrameAnalysis]
