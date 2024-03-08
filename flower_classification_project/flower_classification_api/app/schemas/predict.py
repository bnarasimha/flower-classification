from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from flower_classification_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "SepalLengthCm": "5.1", # datetime.datetime.strptime("2012-11-05", "%Y-%m-%d"),  
                        "SepalWidthCm": "3.5", 
                        "PetalLengthCm": "1.4",
                        "PetalWidthCm": "0.2", 
                    }
                ]
            }
        }
