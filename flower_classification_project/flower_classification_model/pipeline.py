import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from flower_classification_model.config.core import config

flower_classification_pipe = Pipeline([        
    # Regressor
    ('model_rf', KNeighborsClassifier(n_neighbors = config.model_config.n_estimators))
    ])
