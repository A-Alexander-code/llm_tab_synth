"""Render the audit tables and inspect original CSV tokens; never alter inputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


def table(df: pd.DataFrame) -> str:
    def fmt(x: Any) -> str:
        if isinstance(x, float):
            return '—' if pd.isna(x) else f'{x:.10g}'
        return str(x).replace('|', '\\|')
    return '\n'.join(['| '+' | '.join(df.columns)+' |', '| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--results',required=True,type=Path)
    args=parser.parse_args()
    root=args.root.resolve(); out=args.results.resolve()
    s=pd.read_csv(out.joinpath('summary.csv')); e=pd.read_csv(out.joinpath('extreme_records.csv'))
    c=pd.read_csv(out.joinpath('extreme_contributions.csv')); p=pd.read_csv(out.joinpath('predictors.csv'))
    a=pd.read_csv(out.joinpath('aggregation.csv')); cur=s[s.collection=='current']
    tokens=[]; cache={}
    for r in c.itertuples(index=False):
        if r.source_path not in cache:
            with root.joinpath(r.source_path).open(newline='') as f: cache[r.source_path]=list(csv.DictReader(f))
        token=cache[r.source_path][r.source_row_index][r.predictor]
        tokens.append({'collection':r.collection,'dataset':r.dataset,'seed':r.seed,'rank':r.rank,'source_path':r.source_path,'source_csv_line':r.source_csv_line,'predictor':r.predictor,'original_token':token,'parsed_value':r.synth_original,'float_parse_abs_diff':abs(float(token)-r.synth_original)})
    pd.DataFrame(tokens).to_csv(out.joinpath('original_tokens.csv'),index=False)
    checks=[]
    quality=[]
    for collection in ['current','archived']:
        base=root.joinpath('results') if collection=='current' else root.joinpath('results','old_deprecated')
        for ds in ['heart_framingham','diabetes_prima_indian']:
            for seed in [13,23,42,69,88]:
                frames=[pd.read_csv(base.joinpath('samples','tabula',ds,f'seed_{seed}',f'synthetic_class_{label}.csv')) for label in [0,1]]
                df=pd.concat(frames,ignore_index=True)
                split=root.joinpath('data','processed',ds,f'seed_{seed}')
                names=json.loads(split.joinpath('feature_meta.json').read_text())['dataset']['feature_names']
                for population, frame in [('synthetic',df[names]),('train',pd.DataFrame(np.load(split.joinpath('X_train.npy')),columns=names)),('test',pd.DataFrame(np.load(split.joinpath('X_test.npy')),columns=names))]:
                    for col in names:
                        values=frame[col].to_numpy(dtype=float)
                        variance=float(np.nanvar(values))
                        quality.append({'collection':collection,'dataset':ds,'seed':seed,'population':population,'predictor':col,'dtype':str(frame[col].dtype),'nan_count':int(np.isnan(values).sum()),'inf_count':int(np.isinf(values).sum()),'minimum':float(np.nanmin(values)),'maximum':float(np.nanmax(values)),'variance_ddof0':variance,'constant':bool(variance==0),'small_variance_lt_1e_minus12':bool(0<variance<1e-12)})
                if ds=='diabetes_prima_indian':
                    x=df.pregnancies
                    checks.append({'collection':collection,'dataset':ds,'seed':seed,'rule':'pregnancies: nonnegative integer count; absolute tolerance 1e-6','violations':int(((x<0)|((x-x.round()).abs()>1e-6)).sum())})
    count_frame=pd.DataFrame(checks)
    count_frame['violation_pct']=100*count_frame['violations']/1400
    count_frame.to_csv(out.joinpath('count_domain_checks.csv'),index=False)
    pd.DataFrame(quality).to_csv(out.joinpath('data_quality.csv'),index=False)
    manifest=json.loads(out.joinpath('manifest.json').read_text())
    for name in manifest['inputs']:
        if name.endswith('.py') or name.endswith('.ipynb') or name=='config/datasets_description.json':
            dest=out.joinpath('source_snapshot',name); dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(root.joinpath(name).read_bytes())
    extra_source=root.joinpath('src','generators','tabula.py')
    dest=out.joinpath('source_snapshot','src','generators','tabula.py');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(extra_source.read_bytes())
    report=['# Auditoría DCR de TabuLa — IR=1.0',
    '## Resultado principal',
    'La colección actual reproduce las dos cifras mediante la media de cocientes por semilla: Framingham 2.2155010264084673e12 y Pima 3.2241531964610214e9. Framingham también redondea a 2.22e12 con el cociente de medias; Pima distingue ambas fórmulas (3.1597721495214705e9, que redondea a 3.16e9). Las muestras TabuLa de Pima son idénticas en ambas colecciones. Las de Framingham y sus resultados difieren. No se mezclan colecciones.',
    'Los 20 numeradores y denominadores reconstruidos coinciden exactamente con los resultados guardados al leerlos numéricamente; diferencias absolutas y relativas: cero. También coinciden los CSV maestros y por combinación. Esto acredita una reconstrucción numérica de cada colección, no el entorno histórico exacto ni qué archivos se emplearon para elaborar el artículo.',
    'Los denominadores son de orden uno (1.11–1.33). Las cifras enormes proceden de valores ya presentes en los CSV: BMI en Framingham, semillas 42 y 88; pregnancies en Pima, semillas 13, 23 y 88. No se necesitan denominadores pequeños para explicar los cocientes. El recorte diagnóstico devuelve cocientes medios por semilla de 1.033044 en Framingham y 0.893814 en Pima. No reemplaza los resultados originales.',
    '## Método',
    'Se ejecutan build_synthetic_dataset, compute_dcr y compute_dcr_holdout del repositorio, sin entrenar generadores. IR=1 usa las 700 filas de cada clase: 1400 sintéticos por combinación, con selección y barajado originales. Se elimina class y se mantienen nombres/orden. Framingham: 3392 entrenamiento, 848 test, 15 predictores. Pima: 614 entrenamiento, 154 test, 8 predictores. Los arrays reales conservan float32; los CSV usan la inferencia de pandas. StandardScaler se ajusta exclusivamente a entrenamiento, después de fillna(0). BallTree usa euclídea y k=1. No se recodifica, redondea o repara ninguna muestra.',
    'No existe epsilon ni recorte del denominador DCR. El notebook ver3 elimina denominadores no positivos antes de dividir; aquí todos son positivos. El percentil epsilon de k-anonimato no interviene. La media diagnóstica excluye exactamente ceil(0.01*1400)=14 distancias sintéticas mayores por semilla, manteniendo fijo el escalador y denominador. Los percentiles usan el método lineal por defecto de NumPy.',
    '## Resumen por semilla — colección actual',
    table(cur[['dataset','seed','n_train','n_test','n_synth','n_predictors','numerator','denominator','ratio','numerator_abs_diff','numerator_rel_diff','denominator_abs_diff','denominator_rel_diff','ratio_abs_diff','ratio_rel_diff']]),
    '## Distribuciones sintético → entrenamiento',
    table(cur[['dataset','seed']+['synth_'+x for x in ['minimum','median','p90','p95','p99','maximum','mean','zero_count']]]),
    '## Distribuciones test → entrenamiento',
    table(cur[['dataset','seed']+['test_'+x for x in ['minimum','median','p90','p95','p99','maximum','mean','zero_count']]]),
    '## Sensibilidad diagnóstica y registros fuera del rango de entrenamiento',
    table(cur[['dataset','seed','top1_n','top1_distance_sum_share','trimmed_mean','trimmed_ratio','outside_train_rows_n','outside_train_rows_pct']]),
    '## Registros extremos y contribuciones',
    'extreme_records.csv identifica las 14 mayores distancias por combinación: ruta, línea CSV incluyendo cabecera, índice original y vecino real (índice cero de X_train.npy). extreme_contributions.csv contiene todos los predictores, valores originales del sintético y vecino, media/escala, valores estandarizados y aportación a la distancia cuadrática. original_tokens.csv conserva los tokens textuales de esas celdas. Los máximos que originan los cocientes enormes están dominados prácticamente al 100% por BMI o pregnancies.',
    table(e[(e.collection=='current')&(e['rank']==1)][['dataset','seed','source_path','source_csv_line','distance','train_neighbor_index','leading_predictor','leading_squared_share']]),
    table(c[(c.collection=='current')&(c['rank']==1)&(c.squared_share>.5)][['dataset','seed','predictor','synth_original','train_neighbor_original','scaler_scale','squared_share']]),
    'Para los 280 registros extremos de las dos colecciones se contrasta BallTree con scipy cdist contra todos los reales y búsqueda en longdouble. La diferencia máxima de distancia mínima con cdist es cero; el vecino de BallTree tiene brecha cero frente al mejor vecino según longdouble. Esto descarta una discrepancia de búsqueda en estos registros, no prueba ausencia universal de redondeo. No hay desalineación de columnas ni cambio de unidades en el evaluador. Las magnitudes grandes ya están serializadas en los CSV. No hay evidencia que permita atribuir su origen a un mecanismo concreto de TabuLa, a un parser anterior o a una edición posterior.',
    '## Rangos, dominios y restricciones físicas',
    'predictors.csv y predictors.md contienen todos los predictores para cada colección/dataset/semilla: mínimos y máximos de entrenamiento y sintéticos; recuentos y porcentajes por debajo y por encima; códigos inválidos; NaN/inf; constantes y varianzas. Los rangos se ajustan solo a entrenamiento y se comparan exactamente. Fuera del rango observado NO significa inválido. Una fracción binaria dentro de [0,1] puede estar dentro de rango y fuera de dominio.',
    'Para dominios categóricos: binarias {0,1}, education {1,2,3,4}; tolerancia absoluta 1e-6 y relativa cero. Desviaciones no nulas <=1e-6 se separan como precision_only; >1e-6 como substantial. En los datos actuales ninguna infracción categórica es solo precision_only. Pima no tiene predictores categóricos. count_domain_checks.csv comprueba pregnancies como recuento entero no negativo, sin imponer un máximo clínico inventado.',
    'Como restricción física documentada se comprueba BMI>0: BMI es masa positiva dividida por altura positiva al cuadrado. No hay infracciones de esa regla en la colección actual; hay una en Framingham archivado seed_69 (BMI=0). No se impone un umbral clínico superior arbitrario a BMI, edad o pregnancies. Los valores enormes se describen como extremos registrados; su origen y un límite clínico específico no quedan demostrados por este análisis. Las otras restricciones clínicas se marcan not_assessed, no como válidas. Un valor 0 puede además representar un marcador de ausencia, cuya semántica no está registrada en las muestras.',
    'No hay NaN ni infinitos en los datos usados para las distancias. Ningún predictor de entrenamiento es constante ni tiene varianza en (0,1e-12), umbral absoluto diagnóstico declarado. StandardScaler usaría escala 1 para una constante; ese caso no ocurre. Las constantes sintéticas figuran en predictors.csv. data_quality.csv detalla tipos, NaN/inf, constantes y varianzas para entrenamiento, test y sintéticos por separado; small_variance usa el umbral absoluto 1e-12.',
    'Fuentes semánticas: config/datasets_description.json (copia en source_snapshot); [NIDDK, definición de BMI](https://www.niddk.nih.gov/health-information/health-statistics/overweight-obesity); [Framingham/NHLBI, diccionario y unidades](https://biolincc.nhlbi.nih.gov/media/teachingstudies/FHS_Teaching_Longitudinal_Data_Documentation_2021a.pdf). Los dominios documentados no acreditan las unidades realmente utilizadas por un generador al emitir un token.',
    '## Colección archivada — control separado',
    table(s[s.collection=='archived'][['dataset','seed','numerator','denominator','ratio','numerator_abs_diff','denominator_abs_diff','top1_distance_sum_share','trimmed_ratio']]),
    '## Agregación',table(a),
    'El notebook plotting_results_ver3.ipynb contiene build_dcr_seed_ratios y summarize_seed_values: cociente dentro de semilla seguido de media. src/plots/privacy.py y versiones anteriores trabajan con medias agregadas: cociente de medias. El notebook ver3 está guardado con SELECTED_IR=0.8; su código explica el algoritmo pero no acredita por sí solo una ejecución editorial a IR=1. La coincidencia a IR=1 se establece aquí desde resultados e inputs.',
    '## Versiones, hashes y límites de procedencia',
    'HEAD auditado: 2a0325f98514dd6b210e62da28e5dfcf62c13791. Último commit del módulo de privacidad: e619d9d7664da82cfa164f5e728fa7a8491312c9 (2026-03-27). Entorno de reconstrucción: Python y paquetes en manifest.json; no es un entorno histórico certificado. Los hashes SHA256 de entradas y tablas están en manifest.json; los documentos adicionales están en documentation_manifest.json. source_snapshot conserva copias exactas del código. Las entradas se revalidan mediante hash al terminar.',
    'Faltan manifiesto original de evaluación (commit, versiones y hashes de inputs), logs de generación/evaluación, secuencias generadas sin procesar y la vinculación editorial a tablas/figuras del artículo. No se sustituyen los splits guardados por datos raw actuales. El evaluador actual reproduce ambos conjuntos históricos con sus muestras correspondientes y los splits disponibles, pero esa concordancia no prueba que el programa histórico completo fuera idéntico.',
    '## Proposed manuscript text',
    '> At IR=1.0, the mean seed-wise DCR ratios for TabuLa were 2.22×10^12 for Framingham and 3.22×10^9 for Pima. Reconstruction from the stored samples exactly reproduced the saved DCR means. Holdout-to-training mean distances ranged from 1.11 to 1.33; thus the extreme ratios were driven by the synthetic-distance numerator rather than a near-zero reference denominator. A small number of records with exceptionally large BMI values in Framingham and pregnancy counts in Pima dominated these means. As a diagnostic sensitivity analysis, removing the largest 1% of synthetic distances per seed, while retaining the original scaler and denominator, reduced the mean seed-wise ratios to 1.03 and 0.894, respectively. The untrimmed values remain the primary results. This audit establishes sensitivity to extreme recorded values but does not establish their upstream origin.',
    '## Reproducción',
    'Desde la raíz del repositorio, usando las versiones de requirements.txt y una salida que no exista:\n\n```bash\nPYTHONDONTWRITEBYTECODE=1 python audits/tabula_dcr_20260925/audit_dcr.py --root . --out audits/tabula_dcr_20260925/reproduction\nPYTHONDONTWRITEBYTECODE=1 python audits/tabula_dcr_20260925/render_report.py --root . --results audits/tabula_dcr_20260925/reproduction\n```',
    'summary.csv: métricas, distribuciones y concordancia por semilla. predictors.csv/predictors.md: tablas completas por predictor. distances.csv: cada distancia y vecino. extreme_records.csv: top 1%. extreme_contributions.csv: descomposición por predictor. original_tokens.csv: tokens originales. aggregation.csv: dos métodos de agregación. Los archivos distinguen siempre collection.'
    ]
    out.joinpath('REPORT.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    docs=['# Rangos y dominios por predictor y semilla','Porcentajes sobre 1400 sintéticos; referencia: solo entrenamiento. — significa no aplicable/no evaluado. Tolerancia categórica absoluta: 1e-6.']
    for (collection,dataset,seed),g in p.groupby(['collection','dataset','seed']):
        docs += [f'## {collection} / {dataset} / seed {seed}',table(g[['predictor','train_min','train_max','synth_min','synth_max','below_train_n','above_train_n','below_train_pct','above_train_pct','outside_train_pct','domain_exact_pct','domain_precision_only_n','domain_substantial_pct','physical_violation_n']])]
    out.joinpath('predictors.md').write_text('\n\n'.join(docs)+'\n',encoding='utf-8')
    for name,info in manifest['inputs'].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest()==info['sha256'],name
    additional={str(path.relative_to(out)):hashlib.sha256(path.read_bytes()).hexdigest() for path in out.rglob('*') if path.is_file() and path.name!='documentation_manifest.json'}
    additional['render_report.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.joinpath('documentation_manifest.json').write_text(json.dumps(additional,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
