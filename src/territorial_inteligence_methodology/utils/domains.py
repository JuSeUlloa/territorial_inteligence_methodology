from territorial_inteligence_methodology.bronze.config import Config 



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

STRUCTURE_SILVER_TABLE = f"""
WITH agregacion as (
            SELECT 
                            CASE
                WHEN codigo_dane IS NULL THEN NULL
                WHEN LENGTH(CAST(codigo_dane AS STRING)) = 8
                    THEN LEFT(CAST(codigo_dane AS STRING), 5)
                WHEN LENGTH(CAST(codigo_dane AS STRING)) = 5
                    THEN CAST(codigo_dane AS STRING)
                WHEN LENGTH(CAST(codigo_dane AS STRING)) = 4
                    THEN LPAD(CAST(codigo_dane AS STRING),5,'0')
                WHEN LENGTH(CAST(codigo_dane AS STRING)) > 3
                THEN LPAD(
                        LEFT(CAST(codigo_dane AS STRING),LENGTH(CAST(codigo_dane AS STRING)) - 3),5,'0')
                ELSE NULL
            END AS codigo_dane,
                            MAX(municipio)as municipio,
                            MAX(departamento)as departamento,
                            CAST(mes AS INT) as mes,
                            CAST(anio AS INT) as anio,
                            SUM(cantidad) as total_tematica,
                            SUM(CASE WHEN zona = 3 THEN cantidad ELSE 0 END) as tematica_zona_urbana,
                            SUM(CASE WHEN zona = 2 THEN cantidad ELSE 0 END) as tematica_zona_rural,
                            SUM(CASE WHEN zona = 1 THEN cantidad ELSE 0 END) as tematica_zona_otras,
                            SUM(CASE WHEN armas_medios = 3 THEN cantidad ELSE 0 END) as tematica_arma_fuego,
                            SUM(CASE WHEN armas_medios = 2 THEN cantidad ELSE 0 END) as tematica_arma_blanca,
                            SUM(CASE WHEN armas_medios = 4 THEN cantidad ELSE 0 END) as tematica_arma_contundente,
                            SUM(CASE WHEN genero = 2 THEN cantidad ELSE 0 END) as tematica_masculino,
                            SUM(CASE WHEN genero = 1 THEN cantidad ELSE 0 END) as tematica_femenino,
                            SUM(CASE WHEN agrupa_edad = 2 THEN cantidad ELSE 0 END) as tematica_adultos,
                            SUM(CASE WHEN agrupa_edad = 3 THEN cantidad ELSE 0 END) as tematica_jovenes,
                            SUM(CASE WHEN agrupa_edad = 4 THEN cantidad ELSE 0 END) as tematica_menores,
                            SUM(CASE WHEN dia = 'lun.' THEN cantidad ELSE 0 END) as tematica_lunes,
                            SUM(CASE WHEN dia = 'mar.' THEN cantidad ELSE 0 END) as tematica_martes,
                            SUM(CASE WHEN dia = 'mié.' THEN cantidad ELSE 0 END) as tematica_miercoles,
                            SUM(CASE WHEN dia = 'jue.' THEN cantidad ELSE 0 END) as tematica_jueves,
                            SUM(CASE WHEN dia = 'vie.' THEN cantidad ELSE 0 END) as tematica_viernes,
                            SUM(CASE WHEN dia = 'sáb.' THEN cantidad ELSE 0 END) as tematica_sabado,
                            SUM(CASE WHEN dia = 'dom.' THEN cantidad ELSE 0 END) as tematica_domingo
                        FROM {Config().CATALOG}.{Config().SCHEMA}.{Config().TABLE_BRONZE_NAME}
                        WHERE municipio <> '-' AND codigo_dane IS NOT NULL
                        GROUP BY 1, 4, 5
                        ),
                        municipios AS (
                                SELECT
                                    codigo_dane,
                                    MAX(municipio) AS municipio,
                                    MAX(departamento) AS departamento
                                FROM agregacion
                                GROUP BY codigo_dane
                            ),
                        rango_fechas AS (
                                SELECT
                                    MIN(MAKE_DATE(anio, mes, 1)) AS fecha_inicio,
                                    MAX(MAKE_DATE(anio, mes, 1)) AS fecha_fin
                                FROM agregacion
                            ),
                        calendario AS (
                                SELECT
                                    EXPLODE(
                                        SEQUENCE(
                                            fecha_inicio,
                                            fecha_fin,
                                            INTERVAL 1 MONTH
                                        )
                                    ) AS fecha_mes
                                FROM rango_fechas
                            ),

                        periodos AS (
                                SELECT
                                    YEAR(fecha_mes) AS anio,
                                    MONTH(fecha_mes) AS mes
                                FROM calendario
                            ),

                        spine AS (
                                SELECT
                                    m.codigo_dane,
                                    m.municipio,
                                    m.departamento,
                                    p.anio,
                                    p.mes
                                FROM municipios m
                                CROSS JOIN periodos p
                            ),
                        base_completa AS (
                                SELECT
                                    s.codigo_dane,
                                    s.municipio,
                                    s.departamento,
                                    s.anio,
                                    s.mes,
                                    a.total_tematica,
                                    COALESCE(a.total_tematica, 0) AS total_mes,
                                    a.tematica_zona_urbana,
                                    a.tematica_zona_rural,
                                    a.tematica_zona_otras,
                                    a.tematica_arma_fuego,
                                    a.tematica_arma_blanca,
                                    a.tematica_arma_contundente,
                                    a.tematica_masculino,
                                    a.tematica_femenino,
                                    a.tematica_adultos,
                                    a.tematica_jovenes,
                                    a.tematica_menores,
                                    a.tematica_lunes,
                                    a.tematica_martes,
                                    a.tematica_miercoles,
                                    a.tematica_jueves,
                                    a.tematica_viernes,
                                    a.tematica_sabado,
                                    a.tematica_domingo,
                                    CASE
                                        WHEN a.total_tematica IS NOT NULL THEN 1
                                        ELSE 0
                                    END AS hay_observacion
                                FROM spine s
                                LEFT JOIN agregacion a
                                    ON s.codigo_dane = a.codigo_dane
                                   AND s.anio = a.anio
                                   AND s.mes = a.mes
                            ),
                        composicion AS (
                            SELECT
                            b.*,
                            CASE WHEN total_tematica > 0 THEN tematica_zona_urbana / total_tematica * 100 END AS pct_zona_urbana,
                            CASE WHEN total_tematica > 0 THEN tematica_zona_rural / total_tematica * 100 END AS pct_zona_rural,
                            CASE WHEN total_tematica > 0 THEN tematica_zona_otras / total_tematica * 100 END AS pct_zona_otras,
                            CASE WHEN total_tematica > 0 THEN tematica_arma_fuego / total_tematica * 100 END AS pct_arma_fuego,
                            CASE WHEN total_tematica > 0 THEN tematica_arma_blanca / total_tematica * 100 END AS pct_arma_blanca,
                            CASE WHEN total_tematica > 0 THEN tematica_arma_contundente / total_tematica * 100 END AS pct_arma_contundente,
                            CASE WHEN total_tematica > 0 THEN tematica_masculino / total_tematica * 100 END AS pct_genero_masculino,
                            CASE WHEN total_tematica > 0 THEN tematica_femenino / total_tematica * 100 END AS pct_genero_femenino,
                            CASE WHEN total_tematica > 0 THEN tematica_adultos / total_tematica * 100 END AS pct_edad_adultos,
                            CASE WHEN total_tematica > 0 THEN tematica_jovenes / total_tematica * 100 END AS pct_edad_jovenes,
                            CASE WHEN total_tematica > 0 THEN tematica_menores / total_tematica * 100 END AS pct_edad_menores
        FROM base_completa b
    ),
    indexada AS (
        SELECT
            c.*,
            ROW_NUMBER() OVER (
                PARTITION BY codigo_dane
                ORDER BY anio, mes
            ) AS periodo_idx
        FROM composicion c
    ),
    temporal as (
    SELECT
        c.*,

        LAG(total_mes, 1) OVER w AS total_lag_1,
        LAG(total_mes, 2) OVER w AS total_lag_2,
        LAG(total_mes, 3) OVER w AS total_lag_3,
        LAG(total_mes, 4) OVER w AS total_lag_4,
        LAG(total_mes, 5) OVER w AS total_lag_5,
        LAG(total_mes, 6) OVER w AS total_lag_6,
        LAG(total_mes, 12) OVER w AS total_lag_12,
        AVG(total_mes) OVER (
            PARTITION BY codigo_dane ORDER BY anio, mes
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS media_3_raw,
        AVG(total_mes) OVER (
            PARTITION BY codigo_dane ORDER BY anio, mes
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS media_6_raw,
        AVG(total_mes) OVER (
            PARTITION BY codigo_dane ORDER BY anio, mes
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS media_12_raw,
        MAX(total_mes) OVER (
            PARTITION BY codigo_dane ORDER BY anio, mes
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS max_12_meses,
        COUNT(total_tematica) OVER (
            PARTITION BY codigo_dane
        ) AS historia_meses,
        CASE
            WHEN LAG(total_mes, 1) OVER w > 0
            THEN (total_mes - LAG(total_mes, 1) OVER w)
                 / LAG(total_mes, 1) OVER w * 100
        END AS crecimiento_intermensual,

        CASE
            WHEN LAG(total_mes, 12) OVER w > 0
            THEN (total_mes - LAG(total_mes, 12) OVER w)
                 / LAG(total_mes, 12) OVER w * 100
        END AS crecimiento_interanual,

        total_mes - LAG(total_mes, 1) OVER w AS max_mes_anterior,
        total_mes - LAG(total_mes, 12) OVER w AS max_anio_anterior,

        LAG(pct_zona_urbana, 1) OVER w
        AS pct_zona_urbana_lag_1,

        LAG(pct_zona_rural, 1) OVER w
            AS pct_zona_rural_lag_1,

        LAG(pct_arma_fuego, 1) OVER w
            AS pct_arma_fuego_lag_1,

        LAG(pct_arma_blanca, 1) OVER w
            AS pct_arma_blanca_lag_1,

        LAG(pct_arma_contundente, 1) OVER w
            AS pct_arma_contundente_lag_1,

        LAG(pct_genero_masculino, 1) OVER w
            AS pct_genero_masculino_lag_1,

        LAG(pct_genero_femenino, 1) OVER w
            AS pct_genero_femenino_lag_1,

        LAG(pct_edad_adultos, 1) OVER w
            AS pct_edad_adultos_lag_1,

        LAG(pct_edad_jovenes, 1) OVER w
            AS pct_edad_jovenes_lag_1,

        LAG(pct_edad_menores, 1) OVER w
            AS pct_edad_menores_lag_1
    FROM composicion c
    WINDOW w AS (
        PARTITION BY codigo_dane
        ORDER BY anio, mes
    )
    ),
    features AS (
    SELECT
        t.*,
        media_3_raw  AS media_movil_3_meses,
        media_6_raw  AS media_movil_6_meses,
        media_12_raw AS media_movil_12_meses,
        COUNT(total_tematica) OVER (
            PARTITION BY codigo_dane
            ORDER BY anio, mes
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS n_obs_3,

        COUNT(total_tematica) OVER (
            PARTITION BY codigo_dane
            ORDER BY anio, mes
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS n_obs_6,

        COUNT(total_tematica) OVER (
            PARTITION BY codigo_dane
            ORDER BY anio, mes
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS n_obs_12,
        CASE
            WHEN n_obs_3 < 3 THEN 1
            ELSE 0
        END AS historia_insuficiente_3,

        CASE
            WHEN n_obs_6 < 6 THEN 1
            ELSE 0
        END AS historia_insuficiente_6,

        CASE
            WHEN n_obs_12 < 12 THEN 1
            ELSE 0
        END AS historia_insuficiente_12

    FROM temporal t
),

    festividades AS (
        SELECT
            YEAR(date) AS anio,
            MONTH(date) AS mes,
            COUNT(*) AS n_festivos,
            1 AS tiene_festivo
        FROM {Config().CATALOG}.{Config().SCHEMA}.holidays
        GROUP BY
            YEAR(date),
            MONTH(date)
    )
    SELECT
    f.codigo_dane,
    f.municipio,
    f.departamento,
    f.anio,
    f.mes,
    f.total_tematica,
    f.hay_observacion,
    -- variables de rezago
    f.total_lag_1 AS tematica_mes_anterior,
    f.total_lag_1,
    f.total_lag_2,
    f.total_lag_3,
    f.total_lag_4,
    f.total_lag_5,
    f.total_lag_6,
    f.total_lag_12,
    -- Variables móviles
    f.media_movil_3_meses,
    f.media_movil_6_meses,
    f.media_movil_12_meses,
    f.max_12_meses,
    -- Cobertura histórica
    f.n_obs_3,
    f.n_obs_6,
    f.n_obs_12,
    f.historia_meses,
    f.historia_insuficiente_3,
    f.historia_insuficiente_6,
    f.historia_insuficiente_12,
    -- Crecimientos y tendencia
    f.crecimiento_intermensual,
    f.crecimiento_interanual,
    f.max_anio_anterior,
    f.max_mes_anterior,
    -- Composición observada
    f.pct_zona_urbana,
    f.pct_zona_rural,
    f.pct_zona_otras,
    f.pct_arma_fuego,
    f.pct_arma_blanca,
    f.pct_arma_contundente,
    f.pct_genero_masculino,
    f.pct_genero_femenino,
    f.pct_edad_adultos,
    f.pct_edad_jovenes,
    f.pct_edad_menores,
    -- Composición del último mes disponible
    f.pct_zona_urbana_lag_1,
    f.pct_zona_rural_lag_1,
    f.pct_arma_fuego_lag_1,
    f.pct_arma_blanca_lag_1,
    f.pct_arma_contundente_lag_1,
    f.pct_genero_masculino_lag_1,
    f.pct_genero_femenino_lag_1,
    f.pct_edad_adultos_lag_1,
    f.pct_edad_jovenes_lag_1,
    f.pct_edad_menores_lag_1,
    -- Festividades para análisis, no para el modelo
    COALESCE(h.n_festivos, 0) AS n_festivos,
    COALESCE(h.tiene_festivo, 0) AS tiene_festivo

FROM features f

LEFT JOIN festividades h
    ON f.anio = h.anio
   AND f.mes = h.mes

ORDER BY
    f.codigo_dane,
    f.anio,
    f.mes
"""