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

        # el controller vive en extensions: las vistas (que no reciben self)
        # lo encuentran via current_app
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
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", "3300")))


app = APP().create_app()   # target de gunicorn: ...app.app:app


if __name__ == "__main__":
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "4500")), debug=debug)
