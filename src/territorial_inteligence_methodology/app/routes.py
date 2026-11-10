from flask import Blueprint, current_app, jsonify

from territorial_inteligence_methodology.app.controller import Controller


class Routes:

    bq = Blueprint("api", __name__)

    def __init__(self):
        self.controller = Controller(ttl_seconds=600)

    @bq.get("/gold")
    def get_gold_table():
        controller = current_app.extensions["controller"]
        payload, status_code = controller.gold_table()
        return jsonify(payload), status_code
    
    @bq.get("/gold/<mes>")
    def get_gold_table_mes(mes):
        controller = current_app.extensions["controller"]
        payload, status_code = controller.gold_month_table(mes)
        return jsonify(payload), status_code
    
    @bq.get("/prediction")
    def get_prediction_table():
        controller = current_app.extensions["controller"]
        payload, status_code = controller.prediction_table()
        return jsonify(payload), status_code
    
    @bq.get("/prediction/neighborhood/current")
    def get_prediction_municipality_current():
        controller = current_app.extensions["controller"]
        payload, status_code = controller.prediction_municipality_current_table()
        return jsonify(payload), status_code
    
    @bq.get("/prediction/neighborhood")
    def get_prediction_municipality():
        controller = current_app.extensions["controller"]
        payload, status_code = controller.prediction_municipality_table()
        return jsonify(payload), status_code
    
    @bq.get("/prediction/neighborhood/<mes>")
    def get_prediction_municipality_month(mes):
        controller = current_app.extensions["controller"]
        payload, status_code = controller.prediction_municipality_month(mes)
        return jsonify(payload), status_code
    
    
    @bq.get("/department")
    def get_department_table():
        controller = current_app.extensions["controller"]
        payload, status_code = controller.department_geo_table()
        return jsonify(payload), status_code
    
    @bq.get("/bronze")
    def get_bronze_table():
        controller = current_app.extensions["controller"]
        payload, status_code = controller.bronze_table()
        return jsonify(payload), status_code

    @bq.get("/neighborhood")
    def get_barrio_table():
        controller = current_app.extensions["controller"]
        payload, status_code = controller.barrio_geo_table()
        return jsonify(payload), status_code

    @bq.get("/department/<codigo_departamento>/municipality")
    def get_municipalities_by_department(codigo_departamento):
        controller = current_app.extensions["controller"]
        payload, status_code = controller.municipalities_by_department_table(codigo_departamento)
        return jsonify(payload), status_code

    @bq.get("/municipality/<codigo_dane>")
    def get_municipality_by_code(codigo_dane):
        controller = current_app.extensions["controller"]
        payload, status_code = controller.municipality_by_code_table(codigo_dane)
        return jsonify(payload), status_code
