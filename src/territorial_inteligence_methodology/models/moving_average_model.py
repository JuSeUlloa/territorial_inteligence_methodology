import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime

from territorial_inteligence_methodology.models.xgboots_model import XGBoostModel
from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor


class MovingAverageModel:

    def __init__(self, random_state: int = 42):
        self.xgb = XGBoostModel(random_state=random_state)  
        self.config = Config()
        self.sql_executor = SqlExecutor()
        self.catalog = self.config.CATALOG
        self.schema = self.config.SCHEMA
        self.model = None
        self.fill_ = None        
        self.mae_train = None

    def prepare_X(self, df):
        pob = pd.to_numeric(df["pob_dane_2025"], errors="coerce").clip(lower=1)
        X = pd.DataFrame({
            "ma3": pd.to_numeric(df["media_movil_3_meses"], errors="coerce") / pob * 1e5,
            "ma6": pd.to_numeric(df["media_movil_6_meses"], errors="coerce") / pob * 1e5,
        })
        return X.replace([np.inf, -np.inf], np.nan)

    def train(self, df_train):
        X = self.prepare_X(df_train)
        y = pd.to_numeric(df_train["target"], errors="coerce")
        mask = X["ma3"].notna() | X["ma6"].notna()
        X, y = X[mask], y[mask]

        self.fill_ = X.mean(numeric_only=True)
        Xf = X.fillna(self.fill_)

        self.model = LinearRegression().fit(
            np.log1p(Xf.clip(lower=0)), np.log1p(y))
        self.mae_train = float(mean_absolute_error(
            y, np.maximum(np.expm1(self.model.predict(np.log1p(Xf.clip(lower=0)))), 0)))
        print("MODE REG-LOG - Regresión de elasticidades sobre las medias móviles")
        print(f"  intercept β0 = {self.model.intercept_:.4f}")
        print(f"  β1 (MA3)     = {self.model.coef_[0]:.4f}")
        print(f"  β2 (MA6)     = {self.model.coef_[1]:.4f}")
        print(f"  MAE train (in-sample, referencial): {self.mae_train:.4f}")
        return self.model

    def predict(self, df_test):
        X = self.prepare_X(df_test).fillna(self.fill_)
        preds = np.maximum(np.expm1(self.model.predict(np.log1p(X.clip(lower=0)))), 0)
        out = df_test[self.xgb.VARS_ID].copy()
        out["tasa_predicha"] = np.round(preds, 4)
        return out

    def load_train_data(self, train_end_year=None):
        return self.xgb.load_train_data(train_end_year=train_end_year)

    def save_model(self):
        name="media_movil_hurtos.pkl"
        model_dir = "./src/territorial_inteligence_methodology/models"
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, name)
        joblib.dump({"version": "ma-reg-log-v1", "model": self.model,
                     "fill": self.fill_}, path)
        print(f"\nModelo MA guardado en: {path}")
        return path

    def load_model(self):
        name="media_movil_hurtos.pkl"
        path = os.path.join("./src/territorial_inteligence_methodology/models", name)
        obj = joblib.load(path)
        self.model, self.fill_ = obj["model"], obj["fill"]
        print(f"Modelo MA cargado desde: {path}")
        return self.model

    def load_test_data(self, test_year=None):
        return self.xgb.load_test_data(test_year=test_year)

    def run(self):
        df_train = self.load_train_data(train_end_year=datetime.now().year - 1)
        df_test = self.load_test_data(test_year=datetime.now().year)
        self.train(df_train)
        self.save_model()
        df_pred = self.predict(df_test)
        print(f"\nPREDICCIONES MA (reg-log) {datetime.now().year}: {len(df_pred)} filas")
        return df_pred


if __name__ == "__main__":
    MovingAverageModel().run()
