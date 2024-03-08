import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from flower_classification_model import __version__ as _version
from flower_classification_model.config.core import config
from flower_classification_model.processing.data_manager import load_pipeline


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikeshare_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    results = {"predictions": None, "version": _version}
      
    predictions = bikeshare_pipe.predict(input_data)
    results = {"predictions": np.floor(predictions), "version": _version}
    print(results)

    return results


if __name__ == "__main__":
    data_in = [[6, 3, 4, 2]]
    make_prediction(input_data = data_in)