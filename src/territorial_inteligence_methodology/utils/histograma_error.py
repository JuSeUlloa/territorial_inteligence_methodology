from statistics import mean, median, stdev

import matplotlib.pyplot as plt

from territorial_inteligence_methodology.bronze.config import Config
from territorial_inteligence_methodology.utils.sql_executor import SqlExecutor

class  HistogramaError:

    def __init__(self):
        self.config =Config()
        self.sql_executor =SqlExecutor()
        self.catalog =self.config.CATALOG
        self.schema =self.config.SCHEMA
        self.table = f"{self.catalog}.{self.schema}.hurtos_personas_consolidado"

    def fetch_data(self) -> list[dict]:
        query = f"""
            SELECT tasa_real, tasa_predicha, error
            FROM {self.table}
            WHERE error IS NOT NULL
              AND tasa_real IS NOT NULL
              AND tasa_predicha IS NOT NULL
        """
        result = self.sql_executor.execute_sql_query(self.catalog, self.schema, query)
        data = [
            {
                "tasa_real": float(row["tasa_real"]),
                "tasa_predicha": float(row["tasa_predicha"]),
                "error": float(row["error"]),
            }
            for row in result["data"]
        ]
        print(f"  {len(data)} registros obtenidos de {self.table}")
        return data

    def plot(self, bins: int = 30, output_path: str = "histograma_errores.png", show: bool = True):
        print("=" * 50)
        print(f"ANÁLISIS DE ERRORES - {self.table}")
        print("=" * 50)

        data = self.fetch_data()
        if not data:
            print("No hay datos de error para graficar.")
            return

        errors = [d["error"] for d in data]
        real = [d["tasa_real"] for d in data]
        pred = [d["tasa_predicha"] for d in data]

        n = len(errors)
        avg = mean(errors)
        med = median(errors)
        std = stdev(errors) if n > 1 else 0.0
        mae = mean(abs(e) for e in errors)
        rmse = mean(e ** 2 for e in errors) ** 0.5

        fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(16, 6))

        ax_hist.hist(errors, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
        ax_hist.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Sin error (0)")
        ax_hist.axvline(avg, color="red", linestyle="-", linewidth=1.5, label=f"Media: {avg:.2f}")
        ax_hist.axvline(med, color="orange", linestyle="-.", linewidth=1.5, label=f"Mediana: {med:.2f}")
        stats_text = "\n".join([
            f"n = {n}",
            f"Media = {avg:.2f}",
            f"Mediana = {med:.2f}",
            f"Desv. est. = {std:.2f}",
            f"MAE = {mae:.2f}",
            f"RMSE = {rmse:.2f}",
        ])
        ax_hist.text(
            0.98, 0.97, stats_text,
            transform=ax_hist.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        ax_hist.set_title("Distribución del error de predicción")
        ax_hist.set_xlabel("Error (tasa_predicha − tasa_real)")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.legend()

        ax_scatter.scatter(real, pred, alpha=0.4, s=18, color="steelblue", edgecolors="none")
        max_val = max(max(real), max(pred))
        ax_scatter.plot(
            [0, max_val], [0, max_val],
            color="red", linestyle="--", linewidth=1.5,
            label="Predicción exacta (45°)",
        )
        ax_scatter.set_title("Tasa real vs tasa predicha")
        ax_scatter.set_xlabel("Tasa real (por 100.000 hab.)")
        ax_scatter.set_ylabel("Tasa predicha (por 100.000 hab.)")
        ax_scatter.legend()

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        print(f"  Gráficos guardados en: {output_path}")

        if show:
            plt.show()
        plt.close(fig)


def run_histogram():
    HistogramaError().plot()


if __name__ == "__main__":
    run_histogram()