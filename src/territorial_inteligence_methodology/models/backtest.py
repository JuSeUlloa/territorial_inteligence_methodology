import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from territorial_inteligence_methodology.models.xgboots_model import XGBoostModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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

    def run_backtest_xg(self):
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
            preds_te = np.maximum(modelo.predict(X_te), 0)
            filas_fold, motivos = self.evaluate_predictions(
                df_te, actuals, preds_te, mae_train,
                f"{anio} (train 2021-{anio-1})")
            if filas_fold:
                filas.extend(filas_fold)
                fallos.append((f"{anio}-xgb", motivos))
        return self.resumen("XGBOOST", filas, fallos)

    def evaluate_predictions(self, df_te, actuals, preds, mae_train, label,
                             apply_gates=True):
        m = self.model
        ev = df_te[m.VARS_ID + ["pob_dane_2025"]].copy()
        ev["pred"] = preds
        for d in (ev, actuals):
            for k in self.KEYS:
                d[k] = d[k].astype(str)
        ev = ev.merge(actuals[self.KEYS + ["target", "evaluable"]],
                      on=self.KEYS, how="left")
        ev["target"] = pd.to_numeric(ev["target"], errors="coerce")
        ev["evaluable"] = pd.to_numeric(ev["evaluable"], errors="coerce").fillna(0)
        ev = ev[ev["target"].notna() & (ev["evaluable"] == 1)].copy()
        if ev.empty:
            print(f"  Fold {label}: sin filas evaluables, se omite")
            return None, []

        g = self.metrics(ev["target"], ev["pred"])
        gap = g["mae"] / mae_train if mae_train > 0 else float("inf")
        filas = [{"fold": label, "scope": "global", "bin": "global",
                  **g, "gap": gap, "mae_train": mae_train}]
        ev["bin"] = pd.cut(pd.to_numeric(ev["pob_dane_2025"]),
                           bins=self.BINS, labels=self.BINS_LABELS)
        for etiqueta, sub in ev.groupby("bin", observed=True):
            b = self.metrics(sub["target"], sub["pred"])
            filas.append({"fold": label, "scope": "bin", "bin": str(etiqueta),
                          **b, "gap": float("nan"), "mae_train": float("nan")})

        motivos = []
        if not np.isnan(g["r2"]) and g["r2"] < self.R2_MIN:
            motivos.append(f"R2 {g['r2']:.3f} < {self.R2_MIN}")
        if gap > self.GAP_MAX:
            motivos.append(f"gap {gap:.2f}x > {self.GAP_MAX}x (sobreajuste)")
        for r in [f for f in filas if f["scope"] == "bin"]:
            cota = max(self.SESGO_REL_MAX * abs(r["target_medio"]),
                       self.SESGO_ABS_MIN)
            if abs(r["sesgo"]) > cota:
                motivos.append(f"sesgo bin {r['bin']} "
                               f"{r['sesgo']:+.2f} > +-{cota:.1f}")

        print(f"\n--- Fold {label} ---")
        print(f"  n={g['n']}  MAE={g['mae']:.3f}  RMSE={g['rmse']:.3f}  "
              f"R2={g['r2']:.3f}  sesgo={g['sesgo']:+.3f}  gap={gap:.2f}x")
        for r in [f for f in filas if f["scope"] == "bin"]:
            rel = (r["sesgo"] / r["target_medio"] * 100
                   if r["target_medio"] else float("nan"))
            print(f"    {r['bin']:>7}  n={r['n']:>6}  MAE={r['mae']:8.3f}  "
                  f"sesgo={r['sesgo']:+8.3f} ({rel:+.1f}%)")
        if not apply_gates:
            print("  Veredicto: — (baseline de referencia, sin gates)")
            return filas, []
        print("  Veredicto:",
              "OK" if not motivos else "FALLA -> " + "; ".join(motivos))
        return filas, motivos


    def run_backtest_rf(self):
        filas, fallos = [], []
        folds = sorted(int(y) for y in self.FOLDS)
        for anio in folds:
            m = self.model
            m.train_end_year = anio - 1
            df_tr = m.load_train_data(train_end_year=anio - 1)
            df_te = m.load_test_data(anio)
            actuals = m.result_query_df(self.build_actual_query(anio, anio - 1))

            X_tr = m.prepare_X(df_tr, None).copy()
            m.train_categories = {
                c: sorted(df_tr[c].astype(str).unique())
                for c in m.CATEGORICAL_VARS}
            for c in m.CATEGORICAL_VARS:                       # RF: códigos, 0 = unseen
                X_tr[c] = pd.Categorical(X_tr[c], categories=m.train_categories[c]).codes + 1
            y_tr = pd.to_numeric(df_tr["target"], errors="coerce")
            rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                       max_features=0.5, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            preds_tr = np.maximum(rf.predict(X_tr), 0)
            X_te = m.prepare_X(df_te, categories=m.train_categories).copy()
            for c in m.CATEGORICAL_VARS:
                X_te[c] = pd.Categorical(X_te[c], categories=m.train_categories[c]).codes + 1
            preds_te = np.maximum(rf.predict(X_te), 0)
            mae_train = float(mean_absolute_error(y_tr, preds_tr))

            f, motivos = self.evaluate_predictions(
                df_te, actuals, preds_te, mae_train, f"RF {anio} (train 2021-{anio-1})")
            if f:
                filas.extend(f)
                fallos.append((f"{anio}-RF", motivos))

        return self.resumen("RANDOM FOREST", filas, fallos)

    def run_backtest_ma(self):
        filas, fallos = [], []
        folds = sorted(int(y) for y in self.FOLDS)
        for anio in folds:
            m = self.model
            m.train_end_year = anio - 1
            df_tr = m.load_train_data(train_end_year=anio - 1)
            df_te = m.load_test_data(anio)
            actuals = m.result_query_df(self.build_actual_query(anio, anio - 1))

            def X_ma(df):
                pob = pd.to_numeric(df["pob_dane_2025"], errors="coerce").clip(lower=1)
                return pd.DataFrame({
                    "ma3": pd.to_numeric(df["media_movil_3_meses"], errors="coerce") / pob * 1e5,
                    "ma6": pd.to_numeric(df["media_movil_6_meses"], errors="coerce") / pob * 1e5,
                })
            X_tr, X_te = X_ma(df_tr), X_ma(df_te)
            fill = X_tr.mean(numeric_only=True)         
            X_tr_f, X_te_f = (X_tr.replace([np.inf, -np.inf], np.nan).fillna(fill),
                              X_te.replace([np.inf, -np.inf], np.nan).fillna(fill))
            y_tr = pd.to_numeric(df_tr["target"], errors="coerce")

            Xl_tr = np.log1p(X_tr_f.clip(lower=0))
            Xl_te = np.log1p(X_te_f.clip(lower=0))
            reg = LinearRegression().fit(Xl_tr, np.log1p(y_tr))
            print(f"  [MA-reg-log] β0={reg.intercept_:.3f} "
                  f"β1(MA3)={reg.coef_[0]:.3f} β2(MA6)={reg.coef_[1]:.3f}")
            preds_tr = np.maximum(np.expm1(reg.predict(Xl_tr)), 0)
            preds_te = np.maximum(np.expm1(reg.predict(Xl_te)), 0)
            mae_train = float(mean_absolute_error(y_tr, preds_tr))

            f, motivos = self.evaluate_predictions(
                df_te, actuals, preds_te, mae_train,
                f"{anio}-MA (train 2021-{anio-1})")
            if f:
                filas.extend(f)
                fallos.append((f"{anio}-MA", motivos))

        return self.resumen("MEDIAS MÓVILES (reg-log)", filas, fallos)

    def run_2026(self):
        m = self.model
        m.train_end_year = 2025
        df_tr = m.load_train_data(train_end_year=2025)
        df_te = m.load_test_data(2026)
        actuals = m.result_query_df(self.build_actual_query(2026, 2025))

        filas, fallos = [], []

        X_tr = m.prepare_X(df_tr, None)
        m.train_categories = {
            c: list(X_tr[c].cat.categories) for c in m.CATEGORICAL_VARS}
        modelo = m.new_model()
        modelo.fit(X_tr, pd.to_numeric(df_tr["target"], errors="coerce"),
                   verbose=False)
        mae_train = float(mean_absolute_error(
            pd.to_numeric(df_tr["target"], errors="coerce"),
            np.maximum(modelo.predict(X_tr), 0)))
        preds_te = np.maximum(
            modelo.predict(m.prepare_X(df_te, categories=m.train_categories)), 0)
        f, motivos = self.evaluate_predictions(
            df_te, actuals, preds_te, mae_train, "APP2026-xgb")
        if f:
            filas.extend(f)
            fallos.append(("APP2026-xgb", motivos))

        X_tr = m.prepare_X(df_tr, None).copy()
        rf_cats = {c: sorted(df_tr[c].astype(str).unique())
                   for c in m.CATEGORICAL_VARS}
        for c in m.CATEGORICAL_VARS:                    # códigos, 0 = unseen
            X_tr[c] = pd.Categorical(X_tr[c], categories=rf_cats[c]).codes + 1
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                   max_features=0.5, random_state=42, n_jobs=-1)
        rf.fit(X_tr, pd.to_numeric(df_tr["target"], errors="coerce"))
        mae_train = float(mean_absolute_error(
            pd.to_numeric(df_tr["target"], errors="coerce"),
            np.maximum(rf.predict(X_tr), 0)))
        X_te = m.prepare_X(df_te, categories=rf_cats).copy()
        for c in m.CATEGORICAL_VARS:
            X_te[c] = pd.Categorical(X_te[c], categories=rf_cats[c]).codes + 1
        preds_te = np.maximum(rf.predict(X_te), 0)
        f, motivos = self.evaluate_predictions(
            df_te, actuals, preds_te, mae_train, "APP2026-RF")
        if f:
            filas.extend(f)
            fallos.append(("APP2026-RF", motivos))

        def X_ma_rate(df, pob):
            p = pd.to_numeric(pob, errors="coerce").clip(lower=1)
            X = pd.DataFrame({
                "ma3": pd.to_numeric(df["media_movil_3_meses"], errors="coerce") / p * 1e5,
                "ma6": pd.to_numeric(df["media_movil_6_meses"], errors="coerce") / p * 1e5,
            })
            return X.replace([np.inf, -np.inf], np.nan)
        X_tr_ma = X_ma_rate(df_tr, df_tr["pob_dane_2025"])
        y_tr = pd.to_numeric(df_tr["target"], errors="coerce")
        fill = X_tr_ma.mean(numeric_only=True)              # media del TRAIN
        Xl_tr = np.log1p(X_tr_ma.fillna(fill).clip(lower=0))
        reg = LinearRegression().fit(Xl_tr, np.log1p(y_tr))
        print(f"  [MA-reg-log] β0={reg.intercept_:.3f} "
              f"β1(MA3)={reg.coef_[0]:.3f} β2(MA6)={reg.coef_[1]:.3f}")
        mae_train = float(mean_absolute_error(
            y_tr, np.maximum(np.expm1(reg.predict(Xl_tr)), 0)))
        X_te_ma = X_ma_rate(df_te, df_te["pob_dane_2025"]).fillna(fill)
        Xl_te = np.log1p(X_te_ma.clip(lower=0))
        preds_te = np.maximum(np.expm1(reg.predict(Xl_te)), 0)
        f, motivos = self.evaluate_predictions(
            df_te, actuals, preds_te, mae_train, "APP2026-MA")
        if f:
            filas.extend(f)
            fallos.append(("APP2026-MA", motivos))

        return self.resumen("APLICACIÓN 2026 (sin publicar tablas)", filas, fallos)

    def run_priorizacion(self, k_list=(50, 100), meses=6, track="app2026"):
        print("=" * 60)
        print(f"PRIORIZACIÓN TERRITORIAL — track {track} · K={list(k_list)} "
              f"· últimos {meses} meses · universo complete-case")
        print("=" * 60)

        m = self.model
        if track == "app2026":
            train_end, test_year = 2025, 2026
        else:
            train_end, test_year = 2024, 2025
        m.train_end_year = train_end
        df_tr = m.load_train_data(train_end_year=train_end)
        df_te = m.load_test_data(test_year=test_year)
        actuals = m.result_query_df(self.build_actual_query(test_year, train_end))

        X_tr = m.prepare_X(df_tr, None)
        m.train_categories = {
            c: list(X_tr[c].cat.categories) for c in m.CATEGORICAL_VARS}
        modelo = m.new_model()
        modelo.fit(X_tr, pd.to_numeric(df_tr["target"], errors="coerce"),
                   verbose=False)
        preds = modelo.predict(m.prepare_X(df_te, categories=m.train_categories))
        pred = pd.DataFrame(preds, columns=["tasa_predicha"]).set_index(
            df_te.index)
        df_te = df_te.copy()
        df_te["tasa_predicha"] = np.maximum(pred, 0)

        pob = pd.to_numeric(df_te["pob_dane_2025"], errors="coerce").clip(lower=1)
        criterios = {
            "xgboost": df_te["tasa_predicha"],
            "persistencia": pd.to_numeric(df_te["total_lag_1"], errors="coerce") / pob * 1e5,
            "MA3": pd.to_numeric(df_te["media_movil_3_meses"], errors="coerce") / pob * 1e5,
            "MA6": pd.to_numeric(df_te["media_movil_6_meses"], errors="coerce") / pob * 1e5,
        }

        for d in (df_te, actuals):
            for k in self.KEYS:
                d[k] = d[k].astype(str)
        base = df_te[self.KEYS].copy()
        base = base.merge(
            actuals[self.KEYS + ["target", "evaluable"]], on=self.KEYS, how="left")
        base["target"] = pd.to_numeric(base["target"], errors="coerce")
        base["evaluable"] = pd.to_numeric(base["evaluable"], errors="coerce").fillna(0)
        base = base[base["evaluable"] == 1].copy()
        
        base["pob"] = pd.to_numeric(
            df_te.set_index(self.KEYS).reindex(base.set_index(self.KEYS).index)
            ["pob_dane_2025"].to_numpy(), errors="coerce")
        
        crit_frame = pd.concat(
            [s.rename(n) for n, s in criterios.items()], axis=1)
        crit_frame.index = df_te.set_index(self.KEYS).index
        base = base.set_index(self.KEYS)
        base["hurtos_reales"] = (
            base["target"] * base["pob"] / 1e5)
        base = base.join(crit_frame, how="left")

        cols_crit = list(criterios)
        n_antes = len(base)
        base = base[base["target"].notna()
                    & base[cols_crit].notna().all(axis=1)].copy()
        print(f"  universo complete-case: {n_antes} → {len(base)} filas "
              f"({len(base) / n_antes * 100:.1f}%) — idéntico para todos los criterios")
        base = base.reset_index()

        meses_disponibles = sorted(base["mes"].astype(int).unique(),
                                   reverse=True)[:meses]

        filas_salida = []
        for K in k_list:
            resumen_k = {}
            for mes in meses_disponibles:
                mes_f = base[base["mes"].astype(int) == mes]
                n_mes = len(mes_f)
                total_hurtos = mes_f["hurtos_reales"].sum()
                if total_hurtos <= 0:
                    continue
                K_eff = min(K, n_mes)
                top_k_tasa_real = set(
                    mes_f.sort_values("target", ascending=False)
                    .head(K_eff).set_index(self.KEYS).index)
                top_k_count_real = set(
                    mes_f.sort_values("hurtos_reales", ascending=False)
                    .head(K_eff).set_index(self.KEYS).index)
                mean_tasa_universo = mes_f["target"].mean()

                for name in list(criterios) + ["ranking_ideal_conteo", "ranking_ideal_tasa"]:
                    col = name if name in mes_f.columns else None
                    if col is None:
                        scores = (mes_f["hurtos_reales"] if name == "ranking_ideal_conteo"
                                  else mes_f["target"])
                    else:
                        scores = mes_f[col]
                    top = mes_f.assign(_s=scores).sort_values(
                        "_s", ascending=False).head(K_eff)
                    top_ids = set(top.set_index(self.KEYS).index)

                    captura = top["hurtos_reales"].sum() / total_hurtos
                    lift = captura / (K_eff / n_mes)
                    
                    precision_k = len(top_ids & top_k_tasa_real) / K_eff
                    tasa_sel = top["target"].mean()
                    enriquecimiento = (tasa_sel / mean_tasa_universo
                                       if mean_tasa_universo > 0 else np.nan)
                    filas_salida.append({
                        "k": K, "mes": mes, "criterio": name,
                        "n_universo": n_mes, "k_efectivo": K_eff,
                        "hurtos_mes": total_hurtos,
                        "captura": captura, "lift": lift,
                        "precision_k": precision_k,
                        "enriquecimiento": enriquecimiento})
                    resumen_k.setdefault(name, []).append(
                        (captura, lift, precision_k, enriquecimiento))
            print(f"\n─── Top-K = {K} (promedio de los {len(meses_disponibles)} meses) ───")
            print(f"{'criterio':<22} {'captura %':>10} {'lift':>7} "
                  f"{'precision_k tasa %':>14} {'enriq. tasa':>12}")
            for name in list(criterios) + ["ranking_ideal_conteo", "ranking_ideal_tasa"]:
                if name not in resumen_k:
                    continue
                caps = [r[0] for r in resumen_k[name]]
                lifts = [r[1] for r in resumen_k[name]]
                precs = [r[2] for r in resumen_k[name]]
                enrs = [r[3] for r in resumen_k[name]]
                etiqueta = {
                    "ranking_ideal_conteo": "ranking ideal (conteo)",
                    "ranking_ideal_tasa": "ranking ideal (tasa)",
                }.get(name, name)
                print(f"{etiqueta:<22} {100 * float(np.mean(caps)):>9.1f}% "
                      f"{float(np.mean(lifts)):>7.2f}x "
                      f"{100 * float(np.mean(precs)):>13.1f}% "
                      f"{float(np.mean(enrs)):>11.2f}x")

        df_out = pd.DataFrame(filas_salida)
        print("\nDetalle por mes:")
        for K in k_list:
            sub = df_out[df_out["k"] == K]
            for metrica, titulo in [("captura", "captura % (volumen)"),
                                    ("precision_k", "precision_K % (riesgo relativo)")]:
                piv = sub.pivot_table(index="criterio", columns="mes",
                                      values=metrica)
                if not piv.empty:
                    print(f"\nK={K} · {titulo}:")
                    print((piv * 100).round(1).to_string())
        return df_out

    def run_baselines(self):
        filas = []
        for anio in sorted(int(y) for y in self.FOLDS):
            m = self.model
            m.train_end_year = anio - 1
            df_te = m.load_test_data(anio)
            actuals = m.result_query_df(self.build_actual_query(anio, anio - 1))
            pob = pd.to_numeric(df_te["pob_dane_2025"], errors="coerce").clip(lower=1)

            series = {
                "persistencia": pd.to_numeric(df_te["total_lag_1"], errors="coerce"),
                "MA3": pd.to_numeric(df_te["media_movil_3_meses"], errors="coerce"),
                "MA6": pd.to_numeric(df_te["media_movil_6_meses"], errors="coerce"),
            }
            for name, serie in series.items():
                preds = np.maximum(serie / pob * 1e5, 0)
                print(f"\n--- Baseline {name} · fold {anio} ---")
                f, _ = self.evaluate_predictions(
                    df_te, actuals, preds, mae_train=float("nan"),
                    label=f"{name} {anio}", apply_gates=False)
                if f:
                    filas.extend(f)

        df = pd.DataFrame(filas)
        globales = df[df["scope"] == "global"].copy()
        globales["baseline"] = globales["fold"].str.split().str[0]
        print("\n" + "=" * 60)
        print("BASELINES — MAE por fold (gap N/A: sin parámetros entrenados)")
        print(globales.pivot(index="baseline", columns="fold",
                             values="mae").round(3).to_string())
        print("\nLectura: cada baseline aísla una fuente de información")
        print("(persistencia=último mes · MA3=suavizado corto · MA6=nivel estable).")
        print("El valor agregado de las familias entrenadas se mide como su")
        print("mejora sobre estas tres barreras.")
        return {"metrics": df}

    def resumen(self, titulo, filas, fallos):
        ok = all(not m for _, m in fallos)
        print("\n" + "=" * 60)
        print(f"BACKTEST {titulo}:", "APROBADO" if ok else "RECHAZADO")
        for k, m in fallos:
            print(f"  {k}: {'OK' if not m else '; '.join(m)}")
        df = pd.DataFrame(filas)
        print("\nResumen (global por fold):")
        print(df[df["scope"] == "global"][
            ["fold", "n", "mae", "rmse", "r2", "sesgo", "gap"]].to_string(index=False))
        return {"passed": ok, "metrics": df}

if __name__ == "__main__":
    import sys
    test = sys.argv[1] if len(sys.argv) > 1 else "xg"
    bt = Backtest()
    if test == "rf":
        bt.run_backtest_rf()
    elif test == "ma":
        bt.run_backtest_ma()
    elif test == "2026":
        bt.run_2026()
    elif test == "baselines":
        bt.run_baselines()
    elif test == "priorizacion":
        bt.run_priorizacion()
    else:
        bt.run_backtest_xg()
    
   