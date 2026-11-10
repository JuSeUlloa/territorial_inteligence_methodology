# API pública (BFF) para Angular en Vercel

Proxy serverless que expone los datos gold de las dos fases **sin login de
Databricks**: una Vercel Function conecta directa al SQL warehouse con un
service principal (OAuth M2M). La URL de la Databricks App
(`territorial-api`) queda para uso interno (requiere login del workspace).

```
Angular (público) → /api/dbx/<ruta> → api/dbx.mjs (Vercel) → OAuth M2M → Statement Execution → warehouse
```

## Archivos (copiar al proyecto Angular)

| Archivo | Destino |
|---|---|
| `vercel/api/dbx.mjs` | `<angular>/api/dbx.mjs` |
| `vercel.json` | `<angular>/vercel.json` (merge con el existente) |

## Env para Vercel

`Vercel → Project → Settings → Environment Variables` (Production y Preview):

| Variable | Valor |
|---|---|
| `DATABRICKS_HOST` | `dbc-01398c98-b88e.cloud.databricks.com` |
| `DATABRICKS_CLIENT_ID` | `f9ae9d7b-31dd-41a5-83ec-3df38b27dd88` (SP `territorial-api-public`) |
| `DATABRICKS_CLIENT_SECRET` | secret M2M — ver abajo |
| `DATABRICKS_WAREHOUSE_ID` | `56a977392d922752` |
| `DBX_CATALOG` | `workspace` |
| `DBX_SCHEMA` | `territorial_inteligence` |

El secret M2M se creó el 2026-10-28 (expira **2028-10-27**); guardarlo en un
password manager y NO en el repo. Re-deploy después de cambiar los env.

## Rutas soportadas (allow-list SQL)

| GET | Tabla |
|---|---|
| `/api/dbx/gold` · `/api/dbx/gold/1..12` | `hurtos_personas_consolidado` (fase A) |
| `/api/dbx/prediction` | mismo, último mes |
| `/api/dbx/prediction/neighborhood` / `/current` / `/1..12` | `hurto_personas_gold_municipio` (fase B) |

Respuesta: `{ "status": "Ok", "response": [...] }` (igual que la API Flask).

## Consumo desde Angular

```typescript
this.http.get('/api/dbx/prediction/neighborhood/current').subscribe(r => ...)
```

Ruta relativa → mismo origen → cero CORS.

## Permisos aplicados al SP (2026-10-28)

- `USE CATALOG workspace`, `USE SCHEMA workspace.territorial_inteligence`
- `SELECT` en las 24 tablas existentes del schema (no FUTURE TABLES: al crear
  tablas nuevas en el pipeline, re-otorgar con:
  `GRANT SELECT ON TABLE workspace.territorial_inteligence.<tabla> TO \`f9ae9d7b-31dd-41a5-83ec-3df38b27dd88\``)

## Validación end-to-end (hecha 2026-10-28)

Token M2M emitido por `https://<host>/oidc/v1/token` (exp 3600s) y Statement
Execution `POST /api/2.0/sql/statements` → **SUCCEEDED**, top barrios por tasa
predicha 2026-10 (LA DIGNIDAD 122.9, LA CAMPIÑA 103.5, LOS LAURELES 103.4).
