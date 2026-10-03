import os 
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor


class XGBoostModel:

    #variables identificadoras
    VARS_ID=['municipio', 'codigo_dane', 'departamento','mes','anio']

   #variables categoticas
    CATEGORICAL_VARS=[
       'cluster_rv',
       'cluster_pga',
       'cluster_vs',
       'municipio_isla_area_no_municipalizada', 
    ] 
    
    NUM_VARS_SILVER = [
        'mes',
        'anio',
        'pct_zona_urbana',
        'pct_zona_rural',
        'pct_arma_fuego',
        'pct_arma_blanca',
        'pct_arma_contundente',
        'pct_genero_masculino',
        'pct_genero_femenino',
        'pct_edad_adultos',
        'pct_edad_jovenes',
        'pct_edad_menores',
        'pct_dia_lunes',
        'pct_dia_martes',
        'pct_dia_miercoles',
        'pct_dia_jueves',
        'pct_dia_viernes',
        'pct_dia_sabado',
        'pct_dia_domingo',
        'tematica_mes_anterior',
        'media_movil_3_meses', 
        'media_movil_6_meses',
        'tendencia',
        'crecimiento_intermensual',
        'crecimiento_interanual',
        'max_anio_anterior',
        'max_mes_anterior'
    ]

    NUM_VARS_CONSOLIDATED_DATA = [
        'codigo_departamento', 
        'pob_dane_2025',
        'pdet',
        'zomac',
        'pdet_zomac',
        'riesgo_victimizacion',
        'pres_grup_armados',
        'victimizacion_seguridad',
        'pib2024_peso_relativo_mun_valor_agregado_departamental',
        'art95_num1_base_datos_negativa_del_articulo_106_ley_1453_2011',
        'longitud',
        'latitud'
    ]

    FEATURES = NUM_VARS_SILVER + NUM_VARS_CONSOLIDATED_DATA + CATEGORICAL_VARS

    SILVER_COLUMNS="""a.municipio, a.codigo_dane, a.departamento, a.mes, a.anio, a.tematica_mes_anterior,
        a.media_movil_3_meses, a.media_movil_6_meses, a.tendencia,
        a.crecimiento_intermensual, a.crecimiento_interanual,
        a.max_anio_anterior, a.max_mes_anterior,
        a.pct_zona_urbana, a.pct_zona_rural, a.pct_arma_fuego,
        a.pct_arma_blanca, a.pct_arma_contundente, a.pct_genero_masculino,
        a.pct_genero_femenino, a.pct_edad_adultos, a.pct_edad_jovenes,
        a.pct_edad_menores, a.pct_dia_lunes, a.pct_dia_martes,
        a.pct_dia_miercoles, a.pct_dia_jueves, a.pct_dia_viernes,
        a.pct_dia_sabado, a.pct_dia_domingo"""

    CONSOLIDATED_DATA_COLUMNS ="""
    c.codigo_departamento, c.pob_dane_2025, c.pdet, c.zomac, c.pdet_zomac,
    c.riesgo_victimizacion, c.cluster_rv, c.pres_grup_armados,
    c.cluster_pga, c.victimizacion_seguridad, c.cluster_vs,
    c.pib2024_peso_relativo_mun_valor_agregado_departamental,
    c.art95_num1_base_datos_negativa_del_articulo_106_ley_1453_2011,
    c.municipio_isla_area_no_municipalizada, c.longitud, c.latitud
    """

    def __init__(self, random_state=42):
        self.config =Config()
        self.sql_executor =SqlExecutor()
        self.catalog =self.config.CATALOG
        self.schema =self.config.SCHEMA
        self.table_silver =self.config.TABLE_SILVER_NAME
        self.prediction_table =self.config.PREDICTION_TABLE_NAME
        self.catalog =self.config.CATALOG
        self.random_state =random_state
        self.model=None
        self.model_dir ='./src/territorial_inteligence_methodology/models'
        self.train_categories=None

    
    def _base_from(self):
        return (
            f"FROM {self.catalog}.{self.schema}.{self.table_silver} a "
            f"INNER JOIN {self.catalog}.{self.schema}.consolidado c "
            f"ON a.codigo_dane = c.codigo_municipio"
        )

    def build_train_query(self):
        return f"""
            SELECT {self.SILVER_COLUMNS}, {self.CONSOLIDATED_DATA_COLUMNS},
                   (a.total_tematica / c.pob_dane_2025) * 100000 AS target
            {self._base_from()}
            WHERE a.anio BETWEEN 2021 AND 2024
        """

    def build_test_query(self):
        return f"""
            SELECT {self.SILVER_COLUMNS}, {self.CONSOLIDATED_DATA_COLUMNS}
            {self._base_from()}
            WHERE a.anio IN (2025,2026)
            ORDER BY a.departamento, a.municipio ASC
        """

    def result_query_df(self, query):
        result =self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        return pd.DataFrame(result['data'])

    def load_train_data (self):
        print(f"Carganddo Datos de entrenamiento periodo 2021-2024 ...")
        return self.result_query_df(self.build_train_query())

    def load_test_data (self):
        print(f"Carganddo Datos de prueba periodo 2025 y 2026 ...")
        return self.result_query_df(self.build_test_query())

    def prepare_X(self, df, categories:None):
        X = df[self.FEATURES].copy()
        for col in self.NUM_VARS_SILVER + self.NUM_VARS_CONSOLIDATED_DATA:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        for col in self.CATEGORICAL_VARS:
            X[col] = X[col].astype(str).astype("category")
            if categories and col in categories:
                X[col]=pd.Categorical(X[col], categories=categories[col])
            else:
                X[col] = X[col].astype("category") 
        return X

    def new_model(self):
        return XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            enable_categorical=True,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def print_feature_importance(self, top_n=15):
            imp = pd.DataFrame({
                "feature": self.FEATURES,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)
            print(f"\n--- Top {top_n} features ---")
            print(imp.head(top_n).to_string(index=False))
            return imp
    
    def train(self, df_train):
        X = self.prepare_X(df_train, None)
        self.train_categories = {
        col: list(X[col].cat.categories) for col in self.CATEGORICAL_VARS
        }
        y = pd.to_numeric(df_train["target"], errors="coerce")

        # Holdout temporal: 2024 como validacion (metricas honestas)
        mask_val = df_train["anio"].astype(int) == 2024
        X_tr, y_tr = X[~mask_val], y[~mask_val]
        X_va, y_va = X[mask_val], y[mask_val]

        print("=" * 50)
        print("XGBOOST - ENTRENAMIENTO (validacion 2024)")
        print("=" * 50)
        val_model = self.new_model()
        val_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        self.report_metrics(y_va, val_model.predict(X_va), "Validacion 2024")

        print("\nReentrenando modelo final con 2021-2024...")
        self.model = self.new_model()
        self.model.fit(X, y, verbose=False)
        self.print_feature_importance()
        return self.model

    def report_metrics(self, y_true, y_pred, label):
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n--- Metricas ({label}) ---")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE : {mae:.4f}")
        print(f"  R2  : {r2:.4f}")

    def predict(self, df_test):
        X_test = self.prepare_X(df_test, categories=self.train_categories)
        preds = self.model.predict(X_test)
        out = df_test[self.VARS_ID].copy()
        out["tasa_predicha"] = np.round(preds, 4)
        return out

    def save_model(self, name="xgboost_hurtos.pkl"):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, name)
        joblib.dump(self.model, path)
        print(f"\nModelo guardado en: {path}")
        return path

    
    def write_predictions_table(self, df_pred):
        
        cols = list(df_pred.columns)
        rows = []
        for _, r in df_pred.iterrows():
            vals = ", ".join(self._sql_literal(r[c]) for c in cols)
            rows.append(f"({vals})")
        values = ",\n".join(rows)
        prediction_table = f"{self.catalog}.{self.schema}.{self.prediction_table}"
        query = f"""
            CREATE OR REPLACE TABLE {prediction_table} USING DELTA AS
            SELECT * FROM VALUES\n{values}\nAS t({', '.join(cols)})
        """
        print(f"\nEscribiendo tabla: {prediction_table} ({len(df_pred)} filas)...")
        self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        print("  Tabla creada .....")

    @staticmethod
    def _sql_literal(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(v)
        return "'" + str(v).replace("'", "''") + "'"

    
    def run(self, write_table=True):
        df_train = self.load_train_data()
        df_test = self.load_test_data()

        self.train(df_train)
        df_pred = self.predict(df_test)

        self.save_model()
        
        if write_table:
            self.write_predictions_table(df_pred)

        print("\n" + "=" * 50)
        print(f"PREDICCIONES 2025: {len(df_pred)} filas")
        print("=" * 50)
        print(df_pred.head(10).to_string(index=False))
        return df_pred


def run_gold_model():
    XGBoostModel().run()


if __name__ == "__main__":
    run_gold_model()

