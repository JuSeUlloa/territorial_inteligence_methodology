import io
import os

import joblib

from territorial_inteligence_methodology.utils.databricks_client import DataBricksClient


class ModelStore:
    
    def __init__(self):
        self.config = self._load_config()
        self.client = DataBricksClient().get_client()

    def _load_config(self):
        from territorial_inteligence_methodology.bronze.config import Config
        return Config()

    def _volume_path(self, name: str) -> str:
        return (
            f"/Volumes/{self.config.CATALOG}/{self.config.SCHEMA}/"
            f"{self.config.VOLUME_NAME}/files/models/{name}"
        )

    def save(self, model, name="xgboost_hurtos.pkl"):
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        self.client.files.upload(
            self._volume_path(name), buf, overwrite=True
        )

    def load(self, name="xgboost_hurtos.pkl", local_fallback="./src/territorial_inteligence_methodology/models/xgboost_hurtos.pkl"):
        try:
            resp = self.client.files.download(self._volume_path(name))
            stream = getattr(resp, "contents", None)
            if stream is None:
                stream = resp
            data = stream.read()
            return joblib.load(io.BytesIO(data))
        except Exception:
            if os.path.exists(local_fallback):
                return joblib.load(local_fallback)
            raise
