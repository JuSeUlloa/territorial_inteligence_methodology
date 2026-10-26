import argparse

from territorial_inteligence_methodology.bronze.loader import ParquetLoader
from territorial_inteligence_methodology.bronze.quality import Quality
from territorial_inteligence_methodology.bronze.bronce import BronzeEncoder
from territorial_inteligence_methodology.silver.silver import SilverEncoder
from territorial_inteligence_methodology.models.xgboots_model import XGBoostModel
from territorial_inteligence_methodology.gold.consolidated import Consolidated
from territorial_inteligence_methodology.app.app import APP


def main():
    
    parser = argparse.ArgumentParser(
        description="Territorial Intelligence Methodology - Data Pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["quality", "encode","silver", "model", "consolidated","api", "all"],
        default="all",
        help="Step to execute: quality, encode, silver, model, asi consolidated, or all",
    )
    parser.add_argument("--catalog", default=None, help="Override catalog name")
    parser.add_argument("--schema", default=None, help="Override schema name")
    parser.add_argument("--table", default=None, help="Override table name")
    args = parser.parse_args()

    if args.step in ["quality", "all"]:
        quality = Quality()
        quality.execute_quality_checks(
            table_name=args.table  
        )

    if args.step in ["encode", "all"]:
        encoder = BronzeEncoder()
        encoder.create_encoded_table()

    if args.step in ["silver", "all"]:
        encoder = SilverEncoder()
        encoder.execution_silver_table()

    if args.step in ["model", "all"]:
        model = XGBoostModel()
        model.run()
        model.run_forecast()

    if args.step in ["consolidated", "all"]:
        consolidated = Consolidated()
        consolidated.run_shrinkage(compute_in_python=False)
        consolidated.run()
    
    if args.step in ["api", "all"]:
        API = APP()
        API.run()
        

    


if __name__ == "__main__":
    main()

