from enum import Enum
import json
from utils import from_json_with_enum
from dataclasses import dataclass


class DetectorType(Enum):
    YoloV5 = 0
    SSD = 1
    Mask_R_CNN = 2
    UNKNOWN = 3

@dataclass
class VisioDetectorHttpRequest:
    file_name: str
    source_blob_uri: str
    detector_type: DetectorType

    def to_json_dict(self) -> dict:
        return {
            "fileName": self.name,
            "sourceBlobUri": self.age,
            "detectorType": self.detector_type.value
        }

    def to_json_string(self) -> str:
        return json.dumps(self.to_json_dict())
    
    @classmethod
    def from_json(cls, json_dict):

        json_dict['detectorType']
        return cls(
            file_name=json_dict['fileName'],
            source_blob_uri=json_dict['sourceBlobUri'],
            detector_type=from_json_with_enum(json_dict['detectorType'], DetectorType)
        )
    
# class VisioDetectorHttpRequest:
#     def __init__(self,
#                  file_name: str,
#                  source_uri: str,
#                  detector_type: DetectorType):
#         self.file_name = file_name
#         self.source_uri = source_uri
#         self.detector_type = detector_type

#     @classmethod
#     def from_json(cls, json_str: str):
#         json_dict = json.loads(json_str)
#         return cls.from_json_dict(json_dict)

#     @classmethod
#     def from_json_dict(cls, json_dict):
#         try:
#             # Parse the JSON string into a dictionary
#             if isinstance(json_dict, str):
#                 json_dict = json.loads(json_dict)  # loads() converts JSON string to dictionary
            
#             # Extract values from the dictionary
#             file_name = json_dict.get("fileName")
#             source_uri = json_dict.get("sourceUri")
#             detector_type_str = json_dict.get("detectorType")
            
#             # Ensure all required fields are present
#             if not all((file_name, source_uri, detector_type_str)):
#                 raise ValueError("Incomplete data provided in JSON dictionary")

#             # Convert detector type string to an integer and then create the enum
#             detector_type = DetectorType(int(detector_type_str))
            
#             # Create and return an instance of the class
#             return cls(file_name, source_uri, detector_type)
#         except (KeyError, ValueError) as e:
#             # Handle errors during JSON deserialization
#             raise ValueError(f"Error while converting JSON to VisioDetectorHttpRequest: {e}")

#     def to_json(self):
#         return json.dumps(self.to_json_dict())


#     def to_json_dict(self):
#         return {
#             "fileName": self.file_name,
#             "sourceUri": self.source_uri,
#             "detectorType": self.detector_type.value 
#         }

#     def __str__(self):
#         return f"FileName: {self.file_name}, SourceUri: {self.source_uri}, DetectorType: {self.detector_type}"