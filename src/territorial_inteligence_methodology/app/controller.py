from territorial_inteligence_methodology.app.services import Services

class Controller:
    def __init__(self, ttl_seconds:int):
        self.services = Services(ttl_seconds)

    def gold_table(self):
        try:
           gold_table= self.services.get_gold_consolidated_table()
        except Exception as e:
            return {"status":"degraded", "error": str(e)}, 503
        return {
            "status":"Ok",
            "response": gold_table,
        },200
    
    def gold_month_table(self, mes:int):
        try:
           gold_table= self.services.get_gold_consolidated_month_table(mes)
        except ValueError as e:
                    return {"status":"error", "error": str(e)}, 400
        except Exception as e:
                    return {"status":"degraded", "error": str(e)}, 503
        return {
            "status":"Ok",
            "response": gold_table,
        },200
    
    def prediction_table(self):
        try:
           prediction_table= self.services.get_prediction_table()
        except Exception as e:
            return {"status":"degraded", "error": str(e)}, 503
        return {
            "status":"Ok",
            "response": prediction_table,
        },200
    
    def municipality_geo_table(self):
        try:
           municipality= self.services.get_municipality_geo()
        except Exception as e:
            return {"status":"degraded", "error": str(e)}, 503
        return {
            "status":"Ok",
            "response": municipality,
        },200
    
    def department_geo_table(self):
        try:
           department= self.services.get_department_geo()
        except Exception as e:
            return {"status":"degraded", "error": str(e)}, 503
        return {
            "status":"Ok",
            "response": department,
        },200

    def barrio_geo_table(self):
        try:
           barrio= self.services.get_barrio_geo()
        except Exception as e:
            return {"status":"degraded", "error": str(e)}, 503
        return {
            "status":"Ok",
            "response": barrio,
        },200

    def municipalities_by_department_table(self, codigo_departamento: str):
        try:
           municipios= self.services.get_municipalities_by_department(codigo_departamento)
        except ValueError as e:
            return {"status":"error", "error": str(e)}, 400
        except Exception as e:
            return {"status":"degraded", "error": str(e)}, 503
        return {
            "status":"Ok",
            "response": municipios,
        },200

    def municipality_by_code_table(self, codigo_dane: str):
        try:
           municipio= self.services.get_municipality_by_code(codigo_dane)
        except ValueError as e:
            return {"status":"error", "error": str(e)}, 400
        except Exception as e:
            return {"status":"degraded", "error": str(e)}, 503
        return {
            "status":"Ok",
            "response": municipio,
        },200