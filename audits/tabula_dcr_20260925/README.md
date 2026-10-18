# Auditoría de DCR de TabuLa

Informe completo: [results/REPORT.md](results/REPORT.md).

- [Resumen por colección/dataset/semilla](results/summary.csv)
- [Tablas por predictor](results/predictors.md) y [CSV](results/predictors.csv)
- [Calidad de entrenamiento/test/sintéticos](results/data_quality.csv)
- [Registros extremos](results/extreme_records.csv) y [contribuciones](results/extreme_contributions.csv)
- [Tokens originales](results/original_tokens.csv)
- [Todas las distancias](results/distances.csv)
- [Agregación](results/aggregation.csv)
- [Hashes, entorno y parámetros](results/manifest.json)

Reproducción, desde la raíz del repositorio, con las dependencias de requirements.txt y una salida nueva:

```bash
PYTHONDONTWRITEBYTECODE=1 python audits/tabula_dcr_20260925/audit_dcr.py --root . --out audits/tabula_dcr_20260925/reproduction
PYTHONDONTWRITEBYTECODE=1 python audits/tabula_dcr_20260925/render_report.py --root . --results audits/tabula_dcr_20260925/reproduction
```

Se ajusta exclusivamente el escalador de distancias. No se entrenan generadores ni se alteran entradas. El script de cálculo se niega a sobrescribir un directorio de salida existente. Colecciones current y archived siempre separadas. Índices de arrays y de filas de datos empiezan en cero; source_csv_line incluye la cabecera.
