from sklearn import metrics

from config.core import config
from pipeline import flower_classification_pipe
from flower_classification_model.processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split

data = load_dataset(file_name = config.app_config.training_data_file)

X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

flower_classification_pipe.fit(X, y)

save_pipeline(pipeline_to_persist = flower_classification_pipe)