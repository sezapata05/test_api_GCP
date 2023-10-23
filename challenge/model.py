import pandas as pd
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import numpy as np
from datetime import datetime

class DelayModel:

    def __init__(self):
        # Carga el modelo desde el archivo al inicializar la clase
        # self._model = joblib.load('/Users/sezapata/Documents/Latam/XGBoost Model/challenge/modelo_xgboost.pkl')
        self._model = joblib.load('challenge/modelo_xgboost.pkl')
        

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        if target_column is not None:
            data = self.__apply_variable_engineering(data)
            features = data.drop(columns=[target_column])
            # target = data[target_column]
            target = data[[target_column]]  # Aquí convertimos target en un DataFrame
            return self.encode_categorical_features(features), target
        else:
            return self.encode_categorical_features(data)

    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        # Asegúrate de que las columnas categóricas estén codificadas correctamente
        data_encoded = pd.get_dummies(data, columns=["OPERA", "TIPOVUELO", "MES"])
        # data_encoded = pd.get_dummies(data, columns=["OPERA", "TIPOVUELO", "SIGLADES", "DIANOM"])
        
        # Asegúrate de que todas las columnas necesarias estén presentes
        missing_columns = set(self._model.get_booster().feature_names) - set(data_encoded.columns)
        for column in missing_columns:
            data_encoded[column] = 0  # Agrega columnas faltantes con valores 0
        
        # Reordena las columnas para que coincidan con el modelo entrenado
        data_encoded = data_encoded[self._model.get_booster().feature_names]
        
        return data_encoded

    # def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
    #     x_train, _, y_train, _ = train_test_split(features, target, test_size=0.0)
    #     self._model = XGBClassifier(random_state=1, learning_rate=0.01)
    #     self._model.fit(x_train, y_train)
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        x_train = features
        y_train = target
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0/n_y1
        self._model = XGBClassifier(random_state=42, learning_rate=0.01, scale_pos_weight = scale)
        self._model.fit(x_train, y_train)
        # self._model = XGBClassifier(random_state=1, learning_rate=0.01)
        # self._model.fit(x_train, y_train)

    def predict(self, features: pd.DataFrame) -> List[int]:
        predictions = self._model.predict(features)
        
        return predictions.tolist()

    def __apply_variable_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
        data['min_diff'] = data.apply(self.get_min_diff, axis=1)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
        
        return data

    @staticmethod
    def get_period_day(date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()

        if (date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif (date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif (
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return 'noche'

    @staticmethod
    def is_high_season(fecha):
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

        if ((fecha >= range1_min and fecha <= range1_max) or
            (fecha >= range2_min and fecha <= range2_max) or
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    @staticmethod
    def get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff