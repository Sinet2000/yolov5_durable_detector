import json
from dataclasses import dataclass
from typing import Optional
from .visio_detector_http_request import DetectorType
from utils import string_to_enum
from .enums import PredictionClass

@dataclass
class ImagePredictionResult:
    image_name: str
    detector_type: DetectorType
    prediction: float = 0
    classification: Optional[str] = None
    result_img_name: Optional[str] = None
    result_img_path: Optional[str] = None
    errors: str = None
    has_errors: bool = False
    time_taken: float = 0

    def to_json_dict(self) -> dict:
        return {
            "imageName": self.image_name,
            "resultImgName": self.result_img_name,
            "resultImgPath": self.result_img_path,
            "detectorType": self.detector_type.value,
            "predictionClass":  string_to_enum(PredictionClass, self.classification).value,
            "prediction": self.prediction,
            "errors": self.errors,
            "hasErrors": self.has_errors,
            "timeTaken": self.time_taken
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_json_dict())