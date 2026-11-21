import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from territorial_inteligence_methodology.models.xgboots_model import XGBoostModel
from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor


class RandomForestModel:

    RANDOM_FOREST_PARAMS = dict(
        n_estimators=400,        
        min_samples_leaf=2,      
        max_features=0.5,       
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    def __init__(self, random_state: int = 42):
        self.xgb = XGBoostModel(random_state=random_state)  
        self.config = Config()
        self.sql_executor = SqlExecutor()
        self.catalog = self.config.CATALOG
        self.schema = self.config.SCHEMA
        self.random_state = random_state
        self.model = None
        self.train_categories = None
        self.model_dir = "./src/territorial_inteligence_methodology/models"
        self.evaluation_year = datetime.now().year - 1
        self.train_end_year = self.evaluation_year - 1

    def prepare_X(self, df, categories=None):
        X = self.xgb.prepare_X(df, categories=categories).copy()
        for c in self.xgb.CATEGORICAL_VARS:
            values = X[c].astype(str)
            cats = ([str(v) for v in categories[c]]
                    if categories and c in categories
                    else sorted(values.unique()))
            X[c] = pd.Categorical(values, categories=cats).codes.astype(np.int32) + 1
        return X

    def new_model(self):
        return RandomForestRegressor(
            **self.RANDOM_FOREST_PARAMS,
        )

    def train(self, df_train):
        X = self.prepare_X(df_train, None)
        self.train_categories = {
            c: sorted(df_train[c].astype(str).unique())
            for c in self.xgb.CATEGORICAL_VARS
        }
        y = pd.to_numeric(df_train["target"], errors="coerce")

        validation_year = int(pd.to_numeric(df_train["anio"]).max())
        mask_val = df_train["anio"].astype(int) == validation_year
        X_tr, y_tr = X[~mask_val], y[~mask_val]
        X_va, y_va = X[mask_val], y[mask_val]

        print("=" * 50)
        print(f"RANDOM FOREST - ENTRENAMIENTO (validación {validation_year})")
        print("=" * 50)
        val_model = self.new_model()
        val_model.fit(X_tr, y_tr)
        if len(y_va) > 0:
            self.report_metrics(y_va, np.maximum(val_model.predict(X_va), 0),
                                f"Validación {validation_year}")
        else:
            print(f"Advertencia: no hay filas de validación {validation_year}.")

        print(f"\nReentrenando modelo final con 2021-{validation_year}...")
        self.model = self.new_model()
        self.model.fit(X, y)
        self.print_feature_importance()
        return self.model

    def predict(self, df_test):
        X_te = self.prepare_X(df_test, categories=self.train_categories)
        preds = np.maximum(self.model.predict(X_te), 0)
        out = df_test[self.xgb.VARS_ID].copy()
        out["tasa_predicha"] = np.round(preds, 4)
        return out

    def report_metrics(self, y_true, y_pred, label):
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n--- Métricas ({label}) ---")
        print(f"  RMSE: {rmse:.4f}\n  MAE : {mae:.4f}\n  R2  : {r2:.4f}")

    def print_feature_importance(self, top_n=15):
        imp = (pd.DataFrame({"feature": self.xgb.FEATURES,
                             "importance": self.model.feature_importances_})
               .sort_values("importance", ascending=False))
        print(f"\n--- Top {top_n} features (RF) ---")
        print(imp.head(top_n).to_string(index=False))
        return imp

    def save_model(self, name="randomforest_hurtos.pkl"):
        import os
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, name)
        joblib.dump({"version": "rf-v1", "model": self.model,
                     "categories": self.train_categories}, path)
        print(f"\nModelo RF guardado en: {path}")
        return path

    def load_train_data(self, train_end_year=None):
        return self.xgb.load_train_data(train_end_year=train_end_year)

    def load_test_data(self, test_year=None):
        return self.xgb.load_test_data(test_year=test_year)

    def run(self, write_table=True):
        # Ventana de producción, espejo de XGBoostModel.run():
        # entrenar 2021-2025 (validación interna sobre 2025) y aplicar 2026.
        df_train = self.load_train_data(train_end_year=self.evaluation_year)
        df_test = self.load_test_data(test_year=self.evaluation_year + 1)
        self.train(df_train)
        if df_test is not None and not df_test.empty:
            df_pred = self.predict(df_test)
            print(f"\nPREDICCIONES RF {self.evaluation_year + 1}: {len(df_pred)} filas")
            self.save_model()
            return df_pred
        self.save_model()
        return None


if __name__ == "__main__":
    RandomForestModel().run()