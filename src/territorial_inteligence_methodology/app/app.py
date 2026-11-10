import os
import traceback

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from territorial_inteligence_methodology.app.routes import Routes


class APP:

    def __init__(self):
        self.routes = Routes()

    def create_app(self) -> Flask:
        app = Flask(__name__)
        CORS(app, resources={r"/api/*": {"origins": "*"}})

        app.extensions["controller"] = self.routes.controller
        app.register_blueprint(self.routes.bq, url_prefix="/api/territorial_intelligence")

        @app.errorhandler(HTTPException)
        def http(e):
            return jsonify({"error": e.description}), e.code

        @app.errorhandler(Exception)
        def inesperado(e):
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

        return app

    def run(self):
        app = self.create_app()
        
        port = os.getenv("DATABRICKS_APP_PORT") or os.getenv("PORT", "3300")
        app.run(host="0.0.0.0", port=int(port))


app = APP().create_app()


def main():
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    port = os.getenv("DATABRICKS_APP_PORT") or os.getenv("PORT", "3300")
    app.run(host="0.0.0.0", port=int(port), debug=debug)


if __name__ == "__main__":
    main()
