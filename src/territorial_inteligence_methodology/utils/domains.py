
COLUMNS_MAPPING = {
    "Temática": "tematica",
    "Año": "anio",
    "Mes": "mes",
    "Departamento": "departamento",
    "Código Dane": "codigo_dane",
    "Municipio": "municipio",
    "Armas / Medios": "armas_medios",
    "Agrupa Edad Persona": "agrupa_edad",
    "Género": "genero",
    "Zona": "zona",
    "Clase de Sitio": "clase_sitio",
    "Día": "dia",
    "Cantidad": "cantidad"
}


FIELDS_TO_ENCODE = ['genero', 'zona', 'armas_medios', 'agrupa_edad']

FIELDS_NOT_ENCODED = ['tematica', 'anio', 'departamento', 'codigo_dane', 'municipio', 'clase_sitio', 'dia', 'cantidad']

MONTH_MAPPING = {
        'Enero': 1,
        'Febrero': 2,
        'Marzo': 3,
        'Abril': 4,
        'Mayo': 5,
        'Junio': 6,
        'Julio': 7,
        'Agosto': 8,
        'Septiembre': 9,
        'Octubre': 10,
        'Noviembre': 11,
        'Diciembre': 12
}
