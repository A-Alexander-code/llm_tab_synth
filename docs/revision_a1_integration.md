# A1: integración controlada sin entrenamiento

Rama inspeccionada: `revision/a1-data-preparation`, inicialmente sin cambios locales.
No se encontraron `AGENTS.md` ni una skill local `dataset-pipeline` disponible;
se aplicaron las restricciones explícitas del usuario. No se leyeron archivos de
datos de `data/raw/`, incluido `data/raw/deprecated/`. Las cuatro fuentes originales y la preparación
identificada se conservaron sin modificaciones. No se entrenaron generadores ni
clasificadores y no se ejecutó el benchmark completo.

## Ejecución y entorno comprobados

Identificador único aceptado por el consumidor:
`33a9b4fce40753d94f4a3e2bf0c193f129e8edb5b6a19bb393544f747e2f3543`.

Entorno real: `/home/bperalta/llm_tab_venv/bin/python3`, Python 3.10.16,
NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.5.2, SciPy 1.15.3.
La preparación guardada usó Python 3.12.3. Se verificó su identidad original y
se reprodujeron sus artefactos durante la auditoría en Python 3.10; no se calculó
una nueva preparación ni se sustituyó el fingerprint por el del entorno actual.
`main_prepare_revision_a1.py --verify-only` calcula el destino con la versión
actual de Python y, por tanto, no es el comando adecuado para esta comprobación
entre entornos. Se proporciona un consumidor/verificador específico.

Se cargaron con `np.load(..., allow_pickle=False)` las 80 matrices de las 20
combinaciones. Se comprobaron shapes, dtypes exactos (`X=float64`, `y=int32`),
finitud, orden de columnas, etiquetas, posiciones fuente, filas/grupos disjuntos,
conservación de filas, inventarios SHA256, metadatos y procedencia. Se recargó cada
`preprocessing.json` y se reconstruyeron ambas matrices desde las posiciones
originales, con igualdad exacta NumPy. La auditoría también reprodujo, por separado,
las estadísticas y decisiones de preparación y comparó todos sus bytes guardados.

| Dataset | Seeds verificadas | X_train | X_test | Clases train 0/1 | Clases test 0/1 | Test distintos |
| --- | --- | --- | --- | --- | --- | --- |
| Framingham | 13, 23, 42, 69, 88 | 3392 × 15 | 848 × 15 | 2877/515 | 719/129 | 5 |
| Pima | 13, 23, 42, 69, 88 | 614 × 8 | 154 × 8 | 400/214 | 100/54 | 5 |
| Retinopathy | 13, 23, 42, 69, 88 | 921 × 18 | 230 × 18 | 432/489 | 108/122 | 5 |
| Cervical | 13, 23, 42, 69, 88 | 686 × 33 | 172 × 33 | 642/44 | 161/11 | 5 |

En cada fila, shapes y clases son iguales para las cinco seeds; las observaciones
de test son distintas. El informe JSON registra cada combinación individualmente,
incluidos nombres de columnas, hashes de arrays y conjuntos de test e intersecciones
de filas entre cada pareja de seeds.

## Procedimiento exacto de split

Código vigente, sin cambios, en `src/revision_a1.py`, función `split_groups`:

```python
groups: dict[str, list[int]] = defaultdict(list)
for i, group in enumerate(data.group_ids):
    groups[group].append(i)
rng = random.Random(seed)
test_groups: set[str] = set()
for label in (0, 1):
    candidates = sorted(g for g, ids in groups.items() if data.y[ids[0]] == label)
    rng.shuffle(candidates)
    n = sum(len(groups[g]) for g in candidates)
    wanted = math.floor(n * fraction + 0.5)
    reachable = 1
    predecessors: dict[int, tuple[int, str]] = {}
    for group in candidates:
        size = len(groups[group])
        shifted = reachable << size
        new = shifted & ~reachable
        while new:
            bit = new & -new
            total = bit.bit_length() - 1
            predecessors[total] = (total - size, group)
            new -= bit
        reachable |= shifted
    possible = [i for i in range(1, n) if (reachable >> i) & 1]
    if not possible:
        raise ValueError(f"Class {label} cannot appear on both sides of a group split")
    total = min(possible, key=lambda i: (abs(i - wanted), i))
    while total:
        total, group = predecessors[total]
        test_groups.add(group)
train = [i for i, g in enumerate(data.group_ids) if g not in test_groups]
test = [i for i, g in enumerate(data.group_ids) if g in test_groups]
```

La identidad del grupo es SHA256 de nombres ordenados y predictores originales
normalizados mediante decimales exactos y missing explícito. No incluye el target.
Se comprueba que cada grupo tenga una sola clase; una contradicción detiene la
preparación. El target se usa para esa comprobación de consistencia y para
estratificar, nunca para imputar o seleccionar columnas.

Se ordenan hashes y se barajan con un único `random.Random(seed)`; se procesa
primero clase 0 y después clase 1 con el estado avanzado del mismo RNG. Un
subset-sum por tamaños de grupo conserva el primer predecesor descubierto para
cada suma alcanzable. Se elige la suma más cercana a `floor(n_clase*0.2+0.5)`,
con desempate por menor número de filas, excluyendo 0 y n. Los predecesores
determinan los grupos completos de test. Se conserva el orden fuente de filas.
No se puntúan características, utilidad, privacidad ni rendimiento predictivo.
El barajado modifica qué subconjunto alcanza primero cada suma: en estos datos
se observaron cinco subconjuntos distintos por dataset. No se cambiaron splits.

## Consumidores, rutas y compatibilidad

`main_sampler.py` y `main_evaluator.py` aceptan `--revision-a1-run ID`.
Sin esta opción conservan su modo histórico. En modo revisado, la selección
predeterminada se limita a los cuatro datasets y cinco seeds de la preparación.
No se invocan ni el splitter histórico ni `split_groups` ni `Preprocessor.fit`.
La reproducción completa de preparación ocurre exclusivamente en el auditor.
La transformación con el JSON guardado se usa para comprobar la carga; los
componentes reciben los arrays guardados, sin sustituirlos por la reconstrucción.

Faltas de archivos, hashes, shapes, columnas, etiquetas, configuración o
procedencia incompatibles detienen la ejecución. No existe fallback. Las matrices
en memoria son de solo lectura y se crean copias para consumidores. Se rechazan
targets distintos de `class`, cuotas inválidas y selecciones duplicadas.

Cada identidad experimental incluye run ID, hash del manifiesto, dataset/seed,
modelo, todos los hashes de artefactos (incluido y), configuración completa,
versiones del entorno y hashes del código consumidor/generadores/evaluadores.
No se consulta el registro histórico de modelos. No se reutilizan modelos;
`--force-retrain` no se admite en este modo y las salidas existentes se rechazan.

```text
models/revision_a1/<run>/<dataset>/seed_N/<model>/<experiment_sha256>/
train_model_info/revision_a1/<run>/<dataset>/seed_N/<model>/<experiment_sha256>/
results/revision_a1/<run>/<dataset>/seed_N/<model>/<experiment_sha256>/
  identity.json
  synthetic_class_0.csv
  synthetic_class_1.csv
  samples_manifest.json
  metrics/ir_<ir>.csv
  metrics/ir_<ir>/classifiers/        # solo con --save-classifiers true
  metrics/ir_<ir>/degenerate.csv      # si procede
  metrics/input_hashes.json
```

El evaluador exige la identidad y los hashes de las muestras de esta misma
ejecución antes de usarlas. Aborta ante excepciones y métricas `not_computable`,
en lugar de continuar silenciosamente. No agrega ni borra CSV históricos. Las
salidas de métricas son por combinación; la agregación revisada queda pendiente.
Una ejecución interrumpida conserva su carpeta y no se reanuda automáticamente.

## Qué se probó y qué se simuló

El auditor recorrió las 20 combinaciones para los nueve modelos habilitados:
180 recorridos sampler→CSV→evaluador, con componentes simulados. ForestDiffusion
se inspeccionó como contrato de entrada diagnóstico y se comprobó su bloqueo.
Las fixtures repiten filas de train de forma explícita para ejercer las interfaces;
no son muestras generadas por modelos ni resultados científicos del benchmark.
Los archivos de estas simulaciones se crean en un directorio temporal y se eliminan
al terminar; se conservan sus hashes y los de sus entradas en el informe.

| Componente | Entrada comprobada | Ejecución realizada |
| --- | --- | --- |
| Ocho generadores por clase | X_train filtrado por y_train, orden de columnas conservado, sin target ni IDs | `fit/sample` simulados; valores exactos frente al slice esperado |
| TabPFGen | X_train completo + `__target__=y_train`, sin IDs | `fit/sample` simulados; selección de `__label__` conservada |
| ForestDiffusion | Contrato conjunto histórico con `__target__` | Inspección estática y diagnóstico; ejecución bloqueada |
| Clasificadores TRTR | X_train, y_train, X_test, y_test originales | `_split` real de `compute_ml_efficacy`; ajuste/predicción sustituidos por recorder |
| Clasificadores TSTR | X/y de fixture sintética, mismo X_test/y_test | Construcción de dataset y extracción reales; recorder valida igualdad exacta |
| Privacidad geométrica, MIA, NNAA y exact matches | Solo predictores | Dispatch real, métricas simuladas dentro del recorrido |
| Attribute inference | Frames con `class`, según su protocolo | Dispatch real, métrica simulada dentro del recorrido |
| Holdout DCR y radio k-anonymity | Train/test reales, test como consultas geométricas | Funciones reales y StandardScaler/BallTree reales, por separado |

Se registran SHA256 de shape+dtype+bytes en orden C para cada array y, para cada
DataFrame, nombres/orden y hashes por columna. Los recorders de clasificación
validan ambas rutas contra sus arrays esperados, no solo sus dimensiones.
La detección de columnas degeneradas y el builder de IR también se ejecutan de
forma real. El recorrido de integración usa IR=1.0; los IR adicionales conservan
el builder histórico, pero no se presentaron como recorridos simulados probados.
Los dos CLI en `--dry-run` cargaron las 20 particiones y mostraron los diez modelos,
marcando explícitamente el bloqueo de ForestDiffusion; no importaron ni probaron
las dependencias internas de cada generador.

No hay escalado, clipping, redondeo, conversión a float32 ni `fillna` en las nuevas
interfaces a generadores. Se agrega una etiqueta int32 a cada CSV tras generación
por clase, o tras filtrar la etiqueta generada por TabPFGen. Los CSV se leen con
`float_precision='round_trip'`; pandas puede reconstruir la etiqueta como int64.
La validación exige números finitos, columnas en el orden exacto y cuota exacta;
no repara salidas inválidas. No se llama a `postprocess_synthetic` ni se cargan
especificaciones de tipos desde datos históricos.

Operaciones existentes documentadas, fuera de esa interfaz: el wrapper TabPFGen
convierte internamente X a float32 e y a int32; ForestDiffusion también convierte
su matriz conjunta a float32. Las bibliotecas generativas no se ejecutaron ni
se certifican aquí sus transformaciones internas. En evaluación, `_split` usa
`select_dtypes` y `fillna(0)`, que no cambian estos datos numéricos finitos;
los recorders comprobaron la igualdad de valores. DCR, MIA, NNAA y k-anonymity
aplican StandardScaler ajustado con train; sus `fillna(0)` son igualmente inertes
en las entradas revisadas. Exact matches conserva su definición histórica de
redondeo a seis decimales. Las métricas estadísticas excluyen columnas degeneradas
según el código existente. Estas operaciones no son preprocessing de generadores.

## Coincidencias en Cervical y geometría

En las cinco seeds hay cero patrones compartidos en predictores originales y
cero inmediatamente después de eliminar columnas. Las coincidencias aparecen al
imputar; cada patrón compartido afecta a una fila en cada partición.

| Seed | Patrones compartidos | Filas train/test | Clases train/test afectadas | DCR holdout medio | Radio p25 |
| --- | --- | --- | --- | --- | --- |
| 13 | 2 | 2/2 | 0:2 / 0:2 | 1.2407766172 | 0.3699539266 |
| 23 | 2 | 2/2 | 0:2 / 0:2 | 1.5353180155 | 0.3575867326 |
| 42 | 3 | 3/3 | 0:3 / 0:3 | 1.0954855981 | 0.3449377821 |
| 69 | 2 | 2/2 | 0:2 / 0:2 | 1.4380589547 | 0.4128696001 |
| 88 | 0 | 0/0 | ninguna | 1.2782583581 | 0.3494514412 |

Las distancias son euclídeas después de estandarizar con train. Se obtuvieron
2/2/3/2/0 distancias holdout→train exactamente cero, todas en clase 0. En los otros
tres datasets no hubo patrones compartidos después de procesar. No se alteró
ninguna partición para eliminar coincidencias; estas no demuestran por sí solas
leakage ni identidad de sujetos.

El denominador del gráfico DCR es la media holdout→train. Los ceros la reducen,
pero no la anulan aquí. Manteniendo fijo un hipotético numerador, frente a la media
que excluyese solo esos ceros, el cociente aumentaría por 172/170 en seeds 13, 23,
69 y por 172/169 en seed 42. Esta comparación es algebraica, no una modificación
de la evaluación ni una estimación de DCR sintético. El informe guarda también
media y p25 sin ceros como diagnóstico contrafactual.

`compute_kanonymity` toma el percentil 25 de todas las distancias holdout→train;
si epsilon<=0, usa `1e-6`. Los radios reales son positivos y no activan el fallback.
El código cuenta filas en el radio, no sujetos únicos; duplicados dentro del radio
incrementan el recuento. Se probó además una fixture geométrica de filas constantes:
media DCR=0, radio sustituido por `1e-6`, todos los cinco vecinos contados. Se
ejecutó `build_privacy_table` con denominador cero y numeradores 0 y 1: ambos
cocientes quedan NaN (los infinitos se reemplazan por NaN). DCR no tiene un
denominador epsilon alternativo. No se generaron datos sintéticos en estas pruebas.

## ForestDiffusion y trabajo pendiente

El defecto persiste en `src/generators/forestdiffusion.py`: `fit` concatena
predictores y `__target__` sin `label_y`; `sample` elimina la última columna
cuando viene incluida. `main_sampler.py::_generate_and_save` asigna después
`synth_df['class']=label`. Así se etiquetan dos muestreos no condicionados como si
correspondieran a clases distintas. Integrar particiones no corrige este problema.
El modo revisado rechaza entrenamiento y evaluación de este modelo antes de
crear artefactos. El modo histórico conserva el defecto y su comportamiento.

Propuesta para una revisión separada: ajustar con X e y separados mediante la
API supervisada de la versión instalada, conservando o condicionando la etiqueta
generada, y reunir cuotas por clase con límite de intentos y error por déficit.
Antes de habilitarlo hay que comprobar la firma y semántica de la biblioteca,
formas retornadas, fidelidad entre etiquetas y predictores, clases permitidas,
cuotas, reproducción por seed y recarga del modelo. Una alternativa sería entrenar
por clase, pero cambia su protocolo de comparación y debe decidirse explícitamente.
No se aplicó ninguna de estas opciones ni se entrenó el modelo.

No hay bloqueo del entorno para cargar los datos o calcular su geometría.
Siguen pendientes la validación de comportamiento interno de bibliotecas reales,
la corrección de ForestDiffusion, documentación clínica/temporal de Cervical,
independencia por sujeto, agregación revisada y posterior reejecución experimental.
A1 permanece abierto; A2–A5 no se resuelven mediante esta integración.
El siguiente paso recomendado es decidir y validar el protocolo de ForestDiffusion
y revisar las transformaciones internas antes de autorizar una corrida piloto.

## Archivos y comandos reproducibles

Cambios: `main_sampler.py`, `main_evaluator.py` (entrada explícita y dispatcher
estricto); `src/revision_a1_runtime.py` (carga, contratos, identidad y rutas);
`src/revision_a1_audit.py`, `main_check_revision_a1.py` (evidencia sin entrenamiento);
`tests/test_revision_a1_integration.py` (13 pruebas de integración y fallos);
`docs/revision_a1.md`, este informe y `README.md` (documentación).
No se modificaron configuración, preparación, wrappers de generadores ni splits.

```bash
source /home/bperalta/llm_tab_venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/llm-tab-a1-matplotlib
A1_RUN=33a9b4fce40753d94f4a3e2bf0c193f129e8edb5b6a19bb393544f747e2f3543

python3 -m unittest discover -s tests -p 'test_revision_a1*.py' -v
python3 main_check_revision_a1.py --revision-a1-run "$A1_RUN" \
  --output results/revision_a1_integration/reproduction_01
python3 main_sampler.py --revision-a1-run "$A1_RUN" --dry-run
python3 main_evaluator.py --revision-a1-run "$A1_RUN" --dry-run
git diff --check
```

El directorio `--output` debe ser nuevo. El auditor completo escribe únicamente
evidencia; no habilita entrenamiento. La evidencia de esta revisión está en
`results/revision_a1_integration/verified_20261018/report.json` (carpeta de resultados
ignorada por Git). Incluye hashes entregados a cada componente, código y fuentes,
las 20 verificaciones, 180 simulaciones, geometría real y versiones de dependencias.
Se superaron 24 tests en total: 11 existentes y 13 de integración.
SHA256 del informe JSON:
`4b71c0cf017163dc1e300888a23ad8782040b21fa5226bf441ecef36cf8bb8a1`.
