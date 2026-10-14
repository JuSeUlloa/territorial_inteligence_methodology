import os 
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor


class XGBoostModel:

    VARS_ID=['municipio', 'codigo_dane', 'departamento','mes','anio']

    CATEGORICAL_VARS=[
       'cluster_rv',
       'cluster_pga',
       'cluster_vs',
       'municipio_isla_area_no_municipalizada', 
    ] 
    
    NUM_VARS_SILVER = [
    "mes",
    "anio",
    "total_lag_1",
    "total_lag_2",
    "total_lag_3",
    "total_lag_6",
    "total_lag_12",
    "media_movil_3_meses",
    "media_movil_6_meses",
    "media_movil_12_meses",
    "n_obs_3",
    "n_obs_6",
    "n_obs_12",
    "historia_meses",
    "historia_insuficiente_3",
    "historia_insuficiente_6",
    "historia_insuficiente_12",
    "crecimiento_intermensual",
    "crecimiento_interanual",
    "max_anio_anterior",
    "max_mes_anterior",
    "pct_zona_urbana_lag_1",
    "pct_zona_rural_lag_1",
    "pct_arma_fuego_lag_1",
    "pct_arma_blanca_lag_1",
    "pct_arma_contundente_lag_1",
    "pct_genero_masculino_lag_1",
    "pct_genero_femenino_lag_1",
    "pct_edad_adultos_lag_1",
    "pct_edad_jovenes_lag_1",
    "pct_edad_menores_lag_1"
    ]

    NUM_VARS_CONSOLIDATED_DATA = [
        "codigo_departamento",
        "pob_dane_2025",
        "pdet",
        "zomac",
        "pdet_zomac",
        "riesgo_victimizacion",
        "pres_grup_armados",
        "victimizacion_seguridad",
        "pib2024_peso_relativo_mun_valor_agregado_departamental",
        "art95_num1_base_datos_negativa_del_articulo_106_ley_1453_2011",
        "longitud",
        "latitud",
        * CATEGORICAL_VARS
    ]

    DERIVED_FEATURES = [
        'log_pob_dane'
    ]

    CONSOLIDATED_FEATURES = NUM_VARS_CONSOLIDATED_DATA

    FEATURES = NUM_VARS_SILVER + NUM_VARS_CONSOLIDATED_DATA + DERIVED_FEATURES

    SILVER_COLUMNS= ", ".join(
        f"a.{column}" for column in NUM_VARS_SILVER
    )

    SILVER_ID_COLUMNS = (
        "a.municipio, a.codigo_dane, a.departamento"
    )

    CONSOLIDATED_DATA_COLUMNS = ", ".join(
        f"c.{column}" for column in NUM_VARS_CONSOLIDATED_DATA
    )+ ", LOG1P(CAST(c.pob_dane_2025 AS DOUBLE)) AS log_pob_dane"

    def __init__(self, random_state=42):
        self.config =Config()
        self.sql_executor =SqlExecutor()
        self.catalog =self.config.CATALOG
        self.schema =self.config.SCHEMA
        self.table_silver =self.config.TABLE_SILVER_NAME
        self.prediction_table =self.config.PREDICTION_TABLE_NAME
        self.consolidated_table = self.config.CONSOLIDATED_TABLE_NAME
        self.forecast_table ="hurto_personas_forecast"
        self.catalog =self.config.CATALOG
        self.random_state =random_state
        self.model=None
        self.mes_actual = datetime.now().month
        self.anio_actual = datetime.now().year
        self.evaluation_year = self.anio_actual - 1
        self.model_dir ='./src/territorial_inteligence_methodology/models'
        self.train_categories=None

    
    def _base_from(self):
        return (
            f"FROM {self.catalog}.{self.schema}.{self.table_silver} a "
            f"INNER JOIN {self.catalog}.{self.schema}.{self.consolidated_table} c "
            f"ON a.codigo_dane = c.codigo_municipio"
        )

    def build_train_query(self):
        train_end_year = getattr(
            self,
            "train_end_year",
            self.evaluation_year - 1,
        )
        return f"""
            SELECT
                   {self.SILVER_ID_COLUMNS},
                   {self.SILVER_COLUMNS},
                   {self.CONSOLIDATED_DATA_COLUMNS},
                   (a.total_tematica / c.pob_dane_2025) * 100000 AS target
            {self._base_from()}
             WHERE a.anio BETWEEN 2021 AND {train_end_year}
             AND a.hay_observacion = 1
             AND a.total_tematica IS NOT NULL
             AND c.pob_dane_2025 > 0
         """

    def build_test_query(self, test_year=None):
        test_year = test_year or self.anio_actual
        return f"""
            SELECT
                   {self.SILVER_ID_COLUMNS},
                   {self.SILVER_COLUMNS},
                   {self.CONSOLIDATED_DATA_COLUMNS}
            {self._base_from()}
             WHERE a.anio IN ({test_year})
            AND a.hay_observacion = 1
            AND c.pob_dane_2025 > 0
            ORDER BY a.departamento, a.municipio ASC
        """

    def build_forecast_query(self, year: int):
            silver = (f"{self.catalog}.{self.schema}.{self.table_silver}")
            consolidated = (f"{self.catalog}.{self.schema}.{self.consolidated_table}")
    
            return f"""
                WITH historico AS (
                    SELECT a.*
                    FROM {silver} a
                    WHERE (a.anio < {year}
                           OR (a.anio = {year} AND a.mes < (SELECT MAX(mes)+1 FROM {silver} WHERE anio = {year})))
                ),
                ventanas AS (
                    SELECT
                        h.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY h.codigo_dane
                            ORDER BY h.anio DESC, h.mes DESC
                        ) AS rn,
                        LAG(total_tematica, 5) OVER w AS lag_5,
                        LAG(total_tematica, 11) OVER w AS lag_11,
                        AVG(total_tematica) OVER (
                            PARTITION BY codigo_dane ORDER BY anio, mes
                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                        ) AS media_3_fwd,
                        AVG(total_tematica) OVER (
                            PARTITION BY codigo_dane ORDER BY anio, mes
                            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                        ) AS media_6_fwd,
                        AVG(total_tematica) OVER (
                            PARTITION BY codigo_dane ORDER BY anio, mes
                            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                        ) AS media_12_fwd,
                        COUNT(total_tematica) OVER (
                            PARTITION BY codigo_dane ORDER BY anio, mes
                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                        ) AS n_obs_3_fwd,
                        COUNT(total_tematica) OVER (
                            PARTITION BY codigo_dane ORDER BY anio, mes
                            ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                        ) AS n_obs_6_fwd,
                        COUNT(total_tematica) OVER (
                            PARTITION BY codigo_dane ORDER BY anio, mes
                            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                        ) AS n_obs_12_fwd,
                        COUNT(total_tematica) OVER (
                            PARTITION BY codigo_dane
                        ) AS historia_fwd
                    FROM historico h
                    WINDOW w AS (PARTITION BY codigo_dane ORDER BY anio, mes)
                ),
                ultimo_mes AS (
                    SELECT * FROM ventanas WHERE rn = 1
                )
                SELECT
                    u.municipio,
                    u.codigo_dane,
                    u.departamento,
                    (SELECT MAX(mes)+1 FROM {silver} WHERE anio = {year}) AS mes,
                    {year} AS anio,
                    -- Rezagos de la fila M (u = M-1, por eso el desplazamiento)
                    u.total_tematica AS total_lag_1,
                    u.total_lag_1 AS total_lag_2,
                    u.total_lag_2 AS total_lag_3,
                    u.lag_5 AS total_lag_6,
                    u.lag_11 AS total_lag_12,
                    CASE WHEN u.n_obs_3_fwd = 3
                         THEN u.media_3_fwd END AS media_movil_3_meses,
                    CASE WHEN u.n_obs_6_fwd = 6
                         THEN u.media_6_fwd END AS media_movil_6_meses,
                    CASE WHEN u.n_obs_12_fwd = 12
                         THEN u.media_12_fwd END AS media_movil_12_meses,
                    u.n_obs_3_fwd AS n_obs_3,
                    u.n_obs_6_fwd AS n_obs_6,
                    u.n_obs_12_fwd AS n_obs_12,
                    u.historia_fwd AS historia_meses,
                    IF(u.n_obs_3_fwd < 3, 1, 0) AS historia_insuficiente_3,
                    IF(u.n_obs_6_fwd < 6, 1, 0) AS historia_insuficiente_6,
                    IF(u.n_obs_12_fwd < 12, 1, 0) AS historia_insuficiente_12,
                    u.crecimiento_intermensual,
                    u.crecimiento_interanual,
                    u.total_tematica - u.total_lag_1 AS max_mes_anterior,
                    u.total_tematica - u.total_lag_12 AS max_anio_anterior,
                    u.pct_zona_urbana AS pct_zona_urbana_lag_1,
                    u.pct_zona_rural AS pct_zona_rural_lag_1,
                    u.pct_arma_fuego AS pct_arma_fuego_lag_1,
                    u.pct_arma_blanca AS pct_arma_blanca_lag_1,
                    u.pct_arma_contundente AS pct_arma_contundente_lag_1,
                    u.pct_genero_masculino AS pct_genero_masculino_lag_1,
                    u.pct_genero_femenino AS pct_genero_femenino_lag_1,
                    u.pct_edad_adultos AS pct_edad_adultos_lag_1,
                    u.pct_edad_jovenes AS pct_edad_jovenes_lag_1,
                    u.pct_edad_menores AS pct_edad_menores_lag_1,
                    {", ".join(
                        f"c.{column}"
                        for column in self.CONSOLIDATED_FEATURES
                    )+ ", LOG1P(CAST(c.pob_dane_2025 AS DOUBLE)) AS log_pob_dane"}

                FROM ultimo_mes u
                INNER JOIN {consolidated} c
                    ON u.codigo_dane = c.codigo_municipio
                WHERE c.pob_dane_2025 > 0
            """

    def result_query_df(self, query):
        result =self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        return pd.DataFrame(result['data'])

    def load_train_data(self, train_end_year=None):
        self.train_end_year = train_end_year or (
            self.evaluation_year - 1
        )
        print(
            "Cargando datos de entrenamiento periodo "
            f"2021-{self.train_end_year} ..."
        )
        return self.result_query_df(self.build_train_query())

    def load_test_data(self, test_year=None):
        test_year = test_year or self.evaluation_year
        print(f"Cargando datos de prueba periodo {test_year} ...")
        return self.result_query_df(
            self.build_test_query(test_year)
        )

    def load_forecast_data(self, year: int):
            print(
                f"Cargando variables para pronóstico {year} ..."
            )
    
            return self.result_query_df(
                self.build_forecast_query(year)
            )

    def prepare_X(self, df, categories=None):
        X = df[self.FEATURES].copy()
        numeric_columns = [
            column
            for column in self.FEATURES
            if column not in self.CATEGORICAL_VARS
        ]
        for col in numeric_columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )
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

        validation_year = int(
            pd.to_numeric(df_train["anio"]).max()
        )
        mask_val = (
            df_train["anio"].astype(int) == validation_year
        )
        X_tr, y_tr = X[~mask_val], y[~mask_val]
        X_va, y_va = X[mask_val], y[mask_val]

        print("=" * 50)
        print(
            "XGBOOST - ENTRENAMIENTO "
            f"(validacion {validation_year})"
        )
        print("=" * 50)
        val_model = self.new_model()
        val_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        if len(y_va) > 0:
            self.report_metrics(
                y_va,
                val_model.predict(X_va),
                f"Validacion {validation_year}",
            )
        else:
            print(
                "Advertencia: no hay filas para validación "
                f"del año {validation_year}."
            )

        print(
            "\nReentrenando modelo final con "
            f"2021-{validation_year}..."
        )
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
        preds = np.maximum(preds, 0)
        out = df_test[self.VARS_ID].copy()
        out["tasa_predicha"] = np.round(preds, 4)
        return out

    def save_model(self, name="xgboost_hurtos.pkl"):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, name)
        joblib.dump(self.model, path)
        print(f"\nModelo guardado en: {path}")
        return path

    
    def write_predictions_table(self, df_pred, tag:str):
        
        cols = list(df_pred.columns)
        rows = []
        for _, r in df_pred.iterrows():
            vals = ", ".join(self._sql_literal(r[c]) for c in cols)
            rows.append(f"({vals})")
        values = ",\n".join(rows)
        if tag == "forecast":
            prediction_table = f"{self.catalog}.{self.schema}.{self.forecast_table}"
        else:
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

    def run_forecast(
        self,
        write_table=True,
    ):
        date_now = datetime.now()


        anio = date_now.year
        df_train = self.load_train_data(
            train_end_year=anio - 1,
        )
        df_forecast = self.load_forecast_data(
            anio
        )

        self.train(df_train)

        df_predictions = self.predict(
            df_forecast,
        )

        df_predictions["tasa_real"] = None
        df_predictions["estado"] = "MES_EN_CURSO"

        self.save_model()

        if write_table:
            self.write_predictions_table(
                df_predictions,
                'forecast',
            )

        return df_predictions
    
    def run(self, write_table=True):
        df_train = self.load_train_data(
            train_end_year=self.anio_actual - 1,
        )
        df_test = self.load_test_data(
            test_year=self.anio_actual,
        )

        if df_test.empty:
            print(
                f"No hay datos observados para {self.anio_actual}. "
                f"Generando pronóstico de {self.anio_actual}-"
                f"{self.mes_actual:02d}."
            )
            return self.run_forecast(
                write_table=write_table,
            )

        self.train(df_train)
        df_pred = self.predict(df_test)

        self.save_model()
        
        if write_table:
            self.write_predictions_table(df_pred, 'prediccion')

        print("\n" + "=" * 50)
        print(
            f"PREDICCIONES {self.anio_actual}: "
            f"{len(df_pred)} filas"
        )
        print("=" * 50)
        print(df_pred.head(10).to_string(index=False))
        return df_pred


def run_gold_model():
    XGBoostModel().run()
    XGBoostModel().run_forecast()


if __name__ == "__main__":
    run_gold_model()
