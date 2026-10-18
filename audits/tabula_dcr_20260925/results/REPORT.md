# Auditoría DCR de TabuLa — IR=1.0

## Resultado principal

La colección actual reproduce las dos cifras mediante la media de cocientes por semilla: Framingham 2.2155010264084673e12 y Pima 3.2241531964610214e9. Framingham también redondea a 2.22e12 con el cociente de medias; Pima distingue ambas fórmulas (3.1597721495214705e9, que redondea a 3.16e9). Las muestras TabuLa de Pima son idénticas en ambas colecciones. Las de Framingham y sus resultados difieren. No se mezclan colecciones.

Los 20 numeradores y denominadores reconstruidos coinciden exactamente con los resultados guardados al leerlos numéricamente; diferencias absolutas y relativas: cero. También coinciden los CSV maestros y por combinación. Esto acredita una reconstrucción numérica de cada colección, no el entorno histórico exacto ni qué archivos se emplearon para elaborar el artículo.

Los denominadores son de orden uno (1.11–1.33). Las cifras enormes proceden de valores ya presentes en los CSV: BMI en Framingham, semillas 42 y 88; pregnancies en Pima, semillas 13, 23 y 88. No se necesitan denominadores pequeños para explicar los cocientes. El recorte diagnóstico devuelve cocientes medios por semilla de 1.033044 en Framingham y 0.893814 en Pima. No reemplaza los resultados originales.

## Método

Se ejecutan build_synthetic_dataset, compute_dcr y compute_dcr_holdout del repositorio, sin entrenar generadores. IR=1 usa las 700 filas de cada clase: 1400 sintéticos por combinación, con selección y barajado originales. Se elimina class y se mantienen nombres/orden. Framingham: 3392 entrenamiento, 848 test, 15 predictores. Pima: 614 entrenamiento, 154 test, 8 predictores. Los arrays reales conservan float32; los CSV usan la inferencia de pandas. StandardScaler se ajusta exclusivamente a entrenamiento, después de fillna(0). BallTree usa euclídea y k=1. No se recodifica, redondea o repara ninguna muestra.

No existe epsilon ni recorte del denominador DCR. El notebook ver3 elimina denominadores no positivos antes de dividir; aquí todos son positivos. El percentil epsilon de k-anonimato no interviene. La media diagnóstica excluye exactamente ceil(0.01*1400)=14 distancias sintéticas mayores por semilla, manteniendo fijo el escalador y denominador. Los percentiles usan el método lineal por defecto de NumPy.

## Resumen por semilla — colección actual

| dataset | seed | n_train | n_test | n_synth | n_predictors | numerator | denominator | ratio | numerator_abs_diff | numerator_rel_diff | denominator_abs_diff | denominator_rel_diff | ratio_abs_diff | ratio_rel_diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_framingham | 13 | 3392 | 848 | 1400 | 15 | 1.410845601 | 1.281382875 | 1.101033601 | 0 | 0 | 0 | 0 | 0 | 0 |
| heart_framingham | 23 | 3392 | 848 | 1400 | 15 | 1.370914343 | 1.312626236 | 1.044405715 | 0 | 0 | 0 | 0 | 0 | 0 |
| heart_framingham | 42 | 3392 | 848 | 1400 | 15 | 7.864135249e+12 | 1.301113049 | 6.044159847e+12 | 0 | 0 | 0 | 0 | 0 | 0 |
| heart_framingham | 69 | 3392 | 848 | 1400 | 15 | 1.39121817 | 1.327864783 | 1.047710722 | 0 | 0 | 0 | 0 | 0 | 0 |
| heart_framingham | 88 | 3392 | 848 | 1400 | 15 | 6.697852338e+12 | 1.33069598 | 5.033345285e+12 | 0 | 0 | 0 | 0 | 0 | 0 |
| diabetes_prima_indian | 13 | 614 | 154 | 1400 | 8 | 1160140079 | 1.217641314 | 952776540.8 | 0 | 0 | 0 | 0 | 0 | 0 |
| diabetes_prima_indian | 23 | 614 | 154 | 1400 | 8 | 1.779632316e+10 | 1.173303703 | 1.516770391e+10 | 0 | 0 | 0 | 0 | 0 | 0 |
| diabetes_prima_indian | 42 | 614 | 154 | 1400 | 8 | 1.215138318 | 1.222373376 | 0.9940811389 | 0 | 0 | 0 | 0 | 0 | 0 |
| diabetes_prima_indian | 69 | 614 | 154 | 1400 | 8 | 1.091159369 | 1.111274691 | 0.9818988746 | 0 | 0 | 0 | 0 | 0 | 0 |
| diabetes_prima_indian | 88 | 614 | 154 | 1400 | 8 | 364008.8743 | 1.274835467 | 285534.0031 | 0 | 0 | 0 | 0 | 0 | 0 |

## Distribuciones sintético → entrenamiento

| dataset | seed | synth_minimum | synth_median | synth_p90 | synth_p95 | synth_p99 | synth_maximum | synth_mean | synth_zero_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_framingham | 13 | 0.2304180466 | 1.262010257 | 2.175316703 | 2.580432809 | 3.942775559 | 6.743449692 | 1.410845601 | 0 |
| heart_framingham | 23 | 0.2635487984 | 1.209723184 | 2.152693271 | 2.512615157 | 3.540906735 | 7.55229931 | 1.370914343 | 0 |
| heart_framingham | 42 | 0.2689729545 | 1.235928971 | 2.196631203 | 2.647196359 | 3.864471275 | 1.100978935e+16 | 7.864135249e+12 | 0 |
| heart_framingham | 69 | 0.2331996746 | 1.217310626 | 2.15299179 | 2.553291599 | 4.566276829 | 12.84234877 | 1.39121817 | 0 |
| heart_framingham | 88 | 0.3183469458 | 1.241881184 | 2.233555663 | 2.546588491 | 3.691085328 | 9.376993273e+15 | 6.697852338e+12 | 0 |
| diabetes_prima_indian | 13 | 0.2410158843 | 1.043537658 | 1.539616327 | 1.729681831 | 2.44654763 | 1.624196109e+12 | 1160140079 | 0 |
| diabetes_prima_indian | 23 | 0.2744580246 | 1.015693751 | 1.569428115 | 1.769019354 | 2.349473222 | 2.490027761e+13 | 1.779632316e+10 | 0 |
| diabetes_prima_indian | 42 | 0.2442648499 | 1.040091659 | 1.601690067 | 1.777294666 | 2.416017161 | 144.3920148 | 1.215138318 | 0 |
| diabetes_prima_indian | 69 | 0.1992136475 | 1.044614643 | 1.563691633 | 1.777367295 | 2.263362494 | 7.131499373 | 1.091159369 | 0 |
| diabetes_prima_indian | 88 | 0.1996010099 | 1.027225457 | 1.579666024 | 1.771549186 | 2.451797298 | 254805363.6 | 364008.8743 | 0 |

## Distribuciones test → entrenamiento

| dataset | seed | test_minimum | test_median | test_p90 | test_p95 | test_p99 | test_maximum | test_mean | test_zero_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_framingham | 13 | 0.2852185806 | 1.153498542 | 1.861434701 | 2.224872865 | 3.401076177 | 6.052474305 | 1.281382875 | 0 |
| heart_framingham | 23 | 0.2867283092 | 1.180985653 | 1.967229712 | 2.472437116 | 3.300415571 | 6.986139347 | 1.312626236 | 0 |
| heart_framingham | 42 | 0.2864517498 | 1.182351779 | 1.931254886 | 2.474146795 | 3.293691651 | 6.156700046 | 1.301113049 | 0 |
| heart_framingham | 69 | 0.3288025126 | 1.19500109 | 2.029660623 | 2.42188899 | 3.244038646 | 4.404710732 | 1.327864783 | 0 |
| heart_framingham | 88 | 0.4747901611 | 1.211425605 | 1.95898514 | 2.407968631 | 3.298683205 | 4.584744254 | 1.33069598 | 0 |
| diabetes_prima_indian | 13 | 0.4316143137 | 1.076956926 | 1.901864227 | 2.135731254 | 4.104241587 | 5.640148093 | 1.217641314 | 0 |
| diabetes_prima_indian | 23 | 0.332298612 | 1.10844376 | 1.765713181 | 2.093189968 | 2.523771397 | 3.044163443 | 1.173303703 | 0 |
| diabetes_prima_indian | 42 | 0.477469035 | 1.103421369 | 1.819739978 | 2.109077306 | 3.694015696 | 4.75884464 | 1.222373376 | 0 |
| diabetes_prima_indian | 69 | 0.466975994 | 1.072399089 | 1.537530333 | 1.801025412 | 2.585476139 | 4.627043187 | 1.111274691 | 0 |
| diabetes_prima_indian | 88 | 0.4238236882 | 1.128700667 | 1.972976422 | 2.265946265 | 2.961537367 | 3.747186835 | 1.274835467 | 0 |

## Sensibilidad diagnóstica y registros fuera del rango de entrenamiento

| dataset | seed | top1_n | top1_distance_sum_share | trimmed_mean | trimmed_ratio | outside_train_rows_n | outside_train_rows_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heart_framingham | 13 | 14 | 0.03562318246 | 1.374330091 | 1.072536647 | 18 | 1.285714286 |
| heart_framingham | 23 | 14 | 0.0389241528 | 1.330861276 | 1.013892028 | 23 | 1.642857143 |
| heart_framingham | 42 | 14 | 1 | 1.367208688 | 1.050799305 | 31 | 2.214285714 |
| heart_framingham | 69 | 14 | 0.05062799926 | 1.334124826 | 1.004714368 | 15 | 1.071428571 |
| heart_framingham | 88 | 14 | 1 | 1.361673904 | 1.02327949 | 17 | 1.214285714 |
| diabetes_prima_indian | 13 | 14 | 0.9999999991 | 1.070135092 | 0.8788590533 | 49 | 3.5 |
| diabetes_prima_indian | 23 | 14 | 0.9999999999 | 1.058255552 | 0.9019451222 | 46 | 3.285714286 |
| diabetes_prima_indian | 42 | 14 | 0.1165237583 | 1.084389732 | 0.8871182511 | 52 | 3.714285714 |
| diabetes_prima_indian | 69 | 14 | 0.02715634187 | 1.072249971 | 0.9648829223 | 52 | 3.714285714 |
| diabetes_prima_indian | 88 | 14 | 0.9999971005 | 1.06610011 | 0.8362648648 | 50 | 3.571428571 |

## Registros extremos y contribuciones

extreme_records.csv identifica las 14 mayores distancias por combinación: ruta, línea CSV incluyendo cabecera, índice original y vecino real (índice cero de X_train.npy). extreme_contributions.csv contiene todos los predictores, valores originales del sintético y vecino, media/escala, valores estandarizados y aportación a la distancia cuadrática. original_tokens.csv conserva los tokens textuales de esas celdas. Los máximos que originan los cocientes enormes están dominados prácticamente al 100% por BMI o pregnancies.

| dataset | seed | source_path | source_csv_line | distance | train_neighbor_index | leading_predictor | leading_squared_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heart_framingham | 13 | results/samples/tabula/heart_framingham/seed_13/synthetic_class_1.csv | 594 | 6.743449692 | 1479 | prevalentStroke | 0.8935305467 |
| heart_framingham | 23 | results/samples/tabula/heart_framingham/seed_23/synthetic_class_1.csv | 519 | 7.55229931 | 693 | diabetes | 0.6937941126 |
| heart_framingham | 42 | results/samples/tabula/heart_framingham/seed_42/synthetic_class_0.csv | 539 | 1.100978935e+16 | 377 | BMI | 1 |
| heart_framingham | 69 | results/samples/tabula/heart_framingham/seed_69/synthetic_class_1.csv | 630 | 12.84234877 | 2713 | education | 0.9877291601 |
| heart_framingham | 88 | results/samples/tabula/heart_framingham/seed_88/synthetic_class_0.csv | 461 | 9.376993273e+15 | 1827 | BMI | 1 |
| diabetes_prima_indian | 13 | results/samples/tabula/diabetes_prima_indian/seed_13/synthetic_class_0.csv | 661 | 1.624196109e+12 | 607 | pregnancies | 1 |
| diabetes_prima_indian | 23 | results/samples/tabula/diabetes_prima_indian/seed_23/synthetic_class_1.csv | 219 | 2.490027761e+13 | 63 | pregnancies | 1 |
| diabetes_prima_indian | 42 | results/samples/tabula/diabetes_prima_indian/seed_42/synthetic_class_1.csv | 473 | 144.3920148 | 358 | pregnancies | 0.9993865609 |
| diabetes_prima_indian | 69 | results/samples/tabula/diabetes_prima_indian/seed_69/synthetic_class_1.csv | 614 | 7.131499373 | 238 | bloodpressure | 0.8835556469 |
| diabetes_prima_indian | 88 | results/samples/tabula/diabetes_prima_indian/seed_88/synthetic_class_1.csv | 553 | 254805363.6 | 442 | pregnancies | 1 |

| dataset | seed | predictor | synth_original | train_neighbor_original | scaler_scale | squared_share |
| --- | --- | --- | --- | --- | --- | --- |
| heart_framingham | 13 | prevalentStroke | 0.5 | 0 | 0.0784392044 | 0.8935305467 |
| heart_framingham | 23 | diabetes | 0 | 1 | 0.1589664632 | 0.6937941126 |
| heart_framingham | 42 | BMI | 4.531000137e+16 | 56.79999924 | 4.115428546 | 1 |
| heart_framingham | 69 | education | 17 | 4 | 1.018544357 | 0.9877291601 |
| heart_framingham | 88 | BMI | 3.821999931e+16 | 56.79999924 | 4.075933319 | 1 |
| diabetes_prima_indian | 13 | pregnancies | 5.456466675e+12 | 17 | 3.359487592 | 1 |
| diabetes_prima_indian | 23 | pregnancies | 8.461608887e+13 | 17 | 3.398198614 | 1 |
| diabetes_prima_indian | 42 | pregnancies | 495 | 17 | 3.311448217 | 0.9993865609 |
| diabetes_prima_indian | 69 | bloodpressure | 206.8461609 | 122 | 12.65710552 | 0.8835556469 |
| diabetes_prima_indian | 88 | pregnancies | 846160156 | 17 | 3.320809763 | 1 |

Para los 280 registros extremos de las dos colecciones se contrasta BallTree con scipy cdist contra todos los reales y búsqueda en longdouble. La diferencia máxima de distancia mínima con cdist es cero; el vecino de BallTree tiene brecha cero frente al mejor vecino según longdouble. Esto descarta una discrepancia de búsqueda en estos registros, no prueba ausencia universal de redondeo. No hay desalineación de columnas ni cambio de unidades en el evaluador. Las magnitudes grandes ya están serializadas en los CSV. No hay evidencia que permita atribuir su origen a un mecanismo concreto de TabuLa, a un parser anterior o a una edición posterior.

## Rangos, dominios y restricciones físicas

predictors.csv y predictors.md contienen todos los predictores para cada colección/dataset/semilla: mínimos y máximos de entrenamiento y sintéticos; recuentos y porcentajes por debajo y por encima; códigos inválidos; NaN/inf; constantes y varianzas. Los rangos se ajustan solo a entrenamiento y se comparan exactamente. Fuera del rango observado NO significa inválido. Una fracción binaria dentro de [0,1] puede estar dentro de rango y fuera de dominio.

Para dominios categóricos: binarias {0,1}, education {1,2,3,4}; tolerancia absoluta 1e-6 y relativa cero. Desviaciones no nulas <=1e-6 se separan como precision_only; >1e-6 como substantial. En los datos actuales ninguna infracción categórica es solo precision_only. Pima no tiene predictores categóricos. count_domain_checks.csv comprueba pregnancies como recuento entero no negativo, sin imponer un máximo clínico inventado.

Como restricción física documentada se comprueba BMI>0: BMI es masa positiva dividida por altura positiva al cuadrado. No hay infracciones de esa regla en la colección actual; hay una en Framingham archivado seed_69 (BMI=0). No se impone un umbral clínico superior arbitrario a BMI, edad o pregnancies. Los valores enormes se describen como extremos registrados; su origen y un límite clínico específico no quedan demostrados por este análisis. Las otras restricciones clínicas se marcan not_assessed, no como válidas. Un valor 0 puede además representar un marcador de ausencia, cuya semántica no está registrada en las muestras.

No hay NaN ni infinitos en los datos usados para las distancias. Ningún predictor de entrenamiento es constante ni tiene varianza en (0,1e-12), umbral absoluto diagnóstico declarado. StandardScaler usaría escala 1 para una constante; ese caso no ocurre. Las constantes sintéticas figuran en predictors.csv. data_quality.csv detalla tipos, NaN/inf, constantes y varianzas para entrenamiento, test y sintéticos por separado; small_variance usa el umbral absoluto 1e-12.

Fuentes semánticas: config/datasets_description.json (copia en source_snapshot); [NIDDK, definición de BMI](https://www.niddk.nih.gov/health-information/health-statistics/overweight-obesity); [Framingham/NHLBI, diccionario y unidades](https://biolincc.nhlbi.nih.gov/media/teachingstudies/FHS_Teaching_Longitudinal_Data_Documentation_2021a.pdf). Los dominios documentados no acreditan las unidades realmente utilizadas por un generador al emitir un token.

## Colección archivada — control separado

| dataset | seed | numerator | denominator | ratio | numerator_abs_diff | denominator_abs_diff | top1_distance_sum_share | trimmed_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_framingham | 13 | 6.861617353e+10 | 1.281382875 | 5.35485333e+10 | 0 | 0 | 1 | 1.092642052 |
| heart_framingham | 23 | 1.418313903 | 1.312626236 | 1.080516192 | 0 | 0 | 0.05492549043 | 1.031483142 |
| heart_framingham | 42 | 1.481235674 | 1.301113049 | 1.138437336 | 0 | 0 | 0.1178907993 | 1.014369746 |
| heart_framingham | 69 | 1.38493536 | 1.327864783 | 1.042979208 | 0 | 0 | 0.0496326078 | 1.001225687 |
| heart_framingham | 88 | 1.372449007 | 1.33069598 | 1.031376834 | 0 | 0 | 0.07519996019 | 0.9634518557 |
| diabetes_prima_indian | 13 | 1160140079 | 1.217641314 | 952776540.8 | 0 | 0 | 0.9999999991 | 0.8788590533 |
| diabetes_prima_indian | 23 | 1.779632316e+10 | 1.173303703 | 1.516770391e+10 | 0 | 0 | 0.9999999999 | 0.9019451222 |
| diabetes_prima_indian | 42 | 1.215138318 | 1.222373376 | 0.9940811389 | 0 | 0 | 0.1165237583 | 0.8871182511 |
| diabetes_prima_indian | 69 | 1.091159369 | 1.111274691 | 0.9818988746 | 0 | 0 | 0.02715634187 | 0.9648829223 |
| diabetes_prima_indian | 88 | 364008.8743 | 1.274835467 | 285534.0031 | 0 | 0 | 0.9999971005 | 0.8362648648 |

## Agregación

| collection | dataset | mean_seed_ratios | ratio_means | saved_mean_seed_ratios | saved_ratio_means | diagnostic_mean_trimmed_seed_ratios |
| --- | --- | --- | --- | --- | --- | --- |
| archived | diabetes_prima_indian | 3224153196 | 3159772150 | 3224153196 | 3159772150 | 0.8938140427 |
| archived | heart_framingham | 1.070970666e+10 | 1.046986471e+10 | 1.070970666e+10 | 1.046986471e+10 | 1.020634497 |
| current | diabetes_prima_indian | 3224153196 | 3159772150 | 3224153196 | 3159772150 | 0.8938140427 |
| current | heart_framingham | 2.215501026e+12 | 2.221954855e+12 | 2.215501026e+12 | 2.221954855e+12 | 1.033044368 |

El notebook plotting_results_ver3.ipynb contiene build_dcr_seed_ratios y summarize_seed_values: cociente dentro de semilla seguido de media. src/plots/privacy.py y versiones anteriores trabajan con medias agregadas: cociente de medias. El notebook ver3 está guardado con SELECTED_IR=0.8; su código explica el algoritmo pero no acredita por sí solo una ejecución editorial a IR=1. La coincidencia a IR=1 se establece aquí desde resultados e inputs.

## Versiones, hashes y límites de procedencia

HEAD auditado: 2a0325f98514dd6b210e62da28e5dfcf62c13791. Último commit del módulo de privacidad: e619d9d7664da82cfa164f5e728fa7a8491312c9 (2026-03-27). Entorno de reconstrucción: Python y paquetes en manifest.json; no es un entorno histórico certificado. Los hashes SHA256 de entradas y tablas están en manifest.json; los documentos adicionales están en documentation_manifest.json. source_snapshot conserva copias exactas del código. Las entradas se revalidan mediante hash al terminar.

Faltan manifiesto original de evaluación (commit, versiones y hashes de inputs), logs de generación/evaluación, secuencias generadas sin procesar y la vinculación editorial a tablas/figuras del artículo. No se sustituyen los splits guardados por datos raw actuales. El evaluador actual reproduce ambos conjuntos históricos con sus muestras correspondientes y los splits disponibles, pero esa concordancia no prueba que el programa histórico completo fuera idéntico.

## Proposed manuscript text

> At IR=1.0, the mean seed-wise DCR ratios for TabuLa were 2.22×10^12 for Framingham and 3.22×10^9 for Pima. Reconstruction from the stored samples exactly reproduced the saved DCR means. Holdout-to-training mean distances ranged from 1.11 to 1.33; thus the extreme ratios were driven by the synthetic-distance numerator rather than a near-zero reference denominator. A small number of records with exceptionally large BMI values in Framingham and pregnancy counts in Pima dominated these means. As a diagnostic sensitivity analysis, removing the largest 1% of synthetic distances per seed, while retaining the original scaler and denominator, reduced the mean seed-wise ratios to 1.03 and 0.894, respectively. The untrimmed values remain the primary results. This audit establishes sensitivity to extreme recorded values but does not establish their upstream origin.

## Reproducción

Desde la raíz del repositorio, usando las versiones de requirements.txt y una salida que no exista:

```bash
PYTHONDONTWRITEBYTECODE=1 python audits/tabula_dcr_20260925/audit_dcr.py --root . --out audits/tabula_dcr_20260925/reproduction
PYTHONDONTWRITEBYTECODE=1 python audits/tabula_dcr_20260925/render_report.py --root . --results audits/tabula_dcr_20260925/reproduction
```

summary.csv: métricas, distribuciones y concordancia por semilla. predictors.csv/predictors.md: tablas completas por predictor. distances.csv: cada distancia y vecino. extreme_records.csv: top 1%. extreme_contributions.csv: descomposición por predictor. original_tokens.csv: tokens originales. aggregation.csv: dos métodos de agregación. Los archivos distinguen siempre collection.
