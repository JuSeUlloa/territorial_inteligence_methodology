import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from territorial_inteligence_methodology.models.xgboots_model import XGBoostModel
from territorial_inteligence_methodology.bronze.config import Config

class Backtest:

    FOLDS = (2023, 2024, 2025)

    
    R2_MIN = 0.50            
    GAP_MAX = 2.0            
    SESGO_REL_MAX = 0.10      
    SESGO_ABS_MIN = 1.0       
    
    BINS = [0, 10_000, 50_000, 200_000, np.inf]
    
    BINS_LABELS = ["<10k", "10-50k", "50-200k", ">200k"]
    KEYS = ["codigo_dane", "anio", "mes"]

    def __init__(self):
        self.model = XGBoostModel()
        self.config = Config()
        self.catalog = self.config.CATALOG
        self.schema = self.config.SCHEMA
        self.use_thresholds_shrinkage = self.config.USE_THRESHOLDS_SHRINKAGE
        self.prediction_table = self.config.PREDICTION_TABLE_NAME
        self.consolidated_table = self.config.CONSOLIDATED_TABLE_NAME
        self.silver_table = self.config.TABLE_SILVER_NAME
        

    def build_actual_query(self, year, train_end_year=None):
        silver = f"{self.catalog}.{self.schema}.{self.silver_table}"
        consolidated = f"{self.catalog}.{self.schema}.{self.consolidated_table}"

        return f"""
        WITH historico AS (
            SELECT a.codigo_dane,
                   SUM(a.total_tematica) AS casos_hist,
                   COUNT(*) AS meses_ventana
            FROM {silver} a
            WHERE a.anio BETWEEN 2021 AND {train_end_year}
            GROUP BY a.codigo_dane
        )
        SELECT
            a.codigo_dane,
            a.mes,
            a.anio,
            (COALESCE(a.total_tematica, 0) / c.pob_dane_2025) * 100000 AS target,
            CASE
                WHEN a.total_tematica IS NOT NULL
                     OR COALESCE(h.casos_hist, 0) / NULLIF(h.meses_ventana, 0)
                        <= {self.model.LAMBDA_MAX_CERO_PLAUSIBLE}
                THEN 1 ELSE 0
            END AS evaluable
        FROM {silver} a
        INNER JOIN {consolidated} c
            ON a.codigo_dane = c.codigo_municipio
        LEFT JOIN historico h ON a.codigo_dane = h.codigo_dane
        WHERE a.anio = {year}
          AND c.pob_dane_2025 > 0"""

    def metrics(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        r2 = float("nan")
        if len(y_true) > 1 and np.std(y_true) > 0:
            r2 = float(r2_score(y_true, y_pred))
        return {
            "n": int(len(y_true)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": r2,
            "sesgo": float(np.mean(y_pred - y_true)),
            "target_medio": float(np.mean(y_true)),
        }

    def run_backtest(self):
        folds = sorted(int(y) for y in self.FOLDS)
        filas, fallos = [], []
        for anio in folds:
            self.model.train_end_year = anio - 1
            df_tr = self.model.load_train_data(train_end_year=anio - 1)
            df_te = self.model.load_test_data(anio)
            actuals = self.model.result_query_df(self.build_actual_query(anio, anio - 1))

            X_tr = self.model.prepare_X(df_tr, None)
            self.model.train_categories = {
                c: list(X_tr[c].cat.categories) for c in self.model.CATEGORICAL_VARS
            }
            y_tr = pd.to_numeric(df_tr["target"], errors="coerce")
            modelo = self.model.new_model()
            modelo.fit(X_tr, y_tr, verbose=False)
            mae_train = float(
             mean_absolute_error(y_tr, np.maximum(modelo.predict(X_tr), 0))
            )
            X_te = self.model.prepare_X(df_te, categories=self.model.train_categories)
            ev = df_te[self.model.VARS_ID + ["pob_dane_2025"]].copy()
            ev["pred"] = np.maximum(modelo.predict(X_te), 0)
            for d in (ev, actuals):
                for k in self.KEYS:
                    d[k] = d[k].astype(str)
            ev = ev.merge(actuals[self.KEYS + ["target", "evaluable"]], on=self.KEYS, how="left")
            perdidas = int(ev["target"].isna().sum())
            if perdidas:
                print(f"  Aviso: {perdidas} filas sin target en {anio}")
            ev["target"] = pd.to_numeric(ev["target"], errors="coerce")
            ev["evaluable"] = pd.to_numeric(ev["evaluable"], errors="coerce").fillna(0)
            ev = ev[ev["target"].notna() & (ev["evaluable"] == 1)].copy()
            if ev.empty:
                print(f"  Fold {anio}: sin filas evaluables, se omite")
                continue
            g = self.metrics(ev["target"], ev["pred"])
            gap = g["mae"] / mae_train if mae_train > 0 else float("inf")
            filas.append({"fold": anio, "scope": "global", "bin": "global",
                          **g, "gap": gap, "mae_train": mae_train})
            ev["bin"] = pd.cut(
                pd.to_numeric(ev["pob_dane_2025"]), bins=self.BINS, labels=self.BINS_LABELS
            )
            for etiqueta, sub in ev.groupby("bin", observed=True):
                b = self.metrics(sub["target"], sub["pred"])
                filas.append({"fold": anio, "scope": "bin", "bin": str(etiqueta),
                              **b, "gap": float("nan"), "mae_train": float("nan")})
            motivos = []
            if not np.isnan(g["r2"]) and g["r2"] < self.R2_MIN:
                motivos.append(f"R2 {g['r2']:.3f} < {self.R2_MIN}")
            if gap > self.GAP_MAX:
                motivos.append(f"gap {gap:.2f}x > {self.GAP_MAX}x (sobreajuste)")
            for r in [f for f in filas if f["fold"] == anio and f["scope"] == "bin"]:
                cota = max(self.SESGO_REL_MAX * abs(r["target_medio"]), self.SESGO_ABS_MIN)
                if abs(r["sesgo"]) > cota:
                    motivos.append(
                        f"sesgo bin {r['bin']} {r['sesgo']:+.2f} > +-{cota:.1f}"
                    )
            fallos.append((anio, motivos))
            print(f"\n--- Fold {anio} (train 2021-{anio - 1}) ---")
            print(
                f"  n={g['n']}  MAE={g['mae']:.3f}  RMSE={g['rmse']:.3f}  "
                f"R2={g['r2']:.3f}  sesgo={g['sesgo']:+.3f}  gap={gap:.2f}x"
            )
            for r in [f for f in filas if f["fold"] == anio and f["scope"] == "bin"]:
                rel = (r["sesgo"] / r["target_medio"] * 100
                       if r["target_medio"] else float("nan"))
                print(
                    f"    {r['bin']:>7}  n={r['n']:>6}  MAE={r['mae']:8.3f}  "
                    f"sesgo={r['sesgo']:+8.3f} ({rel:+.1f}%)"
                )
            print("  Veredicto:", "OK" if not motivos else "FALLA -> " + "; ".join(motivos))
        ok = all(not motivos for _, motivos in fallos)
        print("\n" + "=" * 60)
        print("BACKTEST", "APROBADO" if ok else "RECHAZADO")
        for anio, motivos in fallos:
            print(f"  {anio}: {'OK' if not motivos else '; '.join(motivos)}")
        return {"passed": ok, "metrics": pd.DataFrame(filas)}

if __name__ == "__main__":
    Backtest().run_backtest()
    
   