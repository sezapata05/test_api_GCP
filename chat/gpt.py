import pandas as pd
import xgboost as xgb
from typing import Tuple, Union, List

class DelayModel:

    def __init__(self):
        self._model = None

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        # Preprocess the data (assuming the data preprocessing code from the previous example)
        features, target = self.preprocess_data(data, target_column)
        
        if target_column is not None:
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Train the XGBoost model
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        # Make predictions using the trained model
        predictions = self._model.predict(features)
        predictions = [1 if y_pred > 0.5 else 0 for y_pred in predictions]
        return predictions

    def preprocess_data(self, data: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Data preprocessing code (assuming the preprocessing code from the previous example)
        # ...
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix='MES')], 
            axis=1
        )
        if target_column is not None:
            target = data[target_column]
            return features, target
        else:
            return features
