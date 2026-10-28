# Detección de tallado de ojos con smartwatch sin fuga entre usuarios

Proyecto refinado para detectar tallado de ojos a partir de señales de smartwatch. El
objetivo actual es binario: A, B y C representan el mismo gesto en diferentes situaciones
y se agrupan como `tallado`; D representa `no_tallado`. La unidad de separación es siempre
el usuario, nunca una ventana individual.

## Principios del experimento

- El ZIP original se lee directamente y no se modifica.
- Cada ventana conserva usuario, etiqueta, archivo, filas, timestamps y calidad.
- Las ventanas se almacenan sin normalizar.
- La normalización se ajusta exclusivamente con el subconjunto de entrenamiento de cada
  partición o fold.
- Los usuarios de test quedan fuera de toda validación cruzada y selección de arquitectura.
- El mejor checkpoint se selecciona por macro-F1 de una validación interna por usuarios.
- El test fijo solo debe evaluarse cuando las arquitecturas estén congeladas.

## Estructura

- `config/`: configuración reproducible.
- `preprocessing/` y `smartwatch_pipeline/`: lectura, control de calidad y datos.
- `models/`: cinco arquitecturas base y dos variantes LSTM separadas.
- `training/`: motor con class weights, early stopping y scheduler.
- `evaluation/`: métricas completas y gráficas.
- `scripts/`: puntos de entrada.
- `data/manifests/`: trazabilidad, calidad y asignación de usuarios.
- `metrics/`, `plots/`, `results/`, `checkpoints/`: artefactos por corrida.

## Preprocesamiento

Desde PowerShell:

```powershell
Set-Location 'C:\Users\olive\smartwatch\modelo_refinado'
& 'C:\Users\olive\anaconda3\Scripts\conda.exe' run -n ctnet python scripts\preprocess.py
& 'C:\Users\olive\anaconda3\Scripts\conda.exe' run -n ctnet python scripts\validate_dataset.py
```

Resultados principales:

- `data/processed/smartwatch_windows.npz`: ventanas crudas y metadatos.
- `data/manifests/windows_raw.csv`: una fila por ventana.
- `data/manifests/file_quality.csv`: auditoría por CSV.
- `data/manifests/windows_with_splits.csv`: holdout y fold asignados.
- `data/manifests/subjects_with_splits.csv`: distribución por usuario.
- `data/manifests/dataset_summary.json` y `split_summary.json`.

El control temporal conserva el orden de adquisición. Repara únicamente timestamps
aislados cuando ambos vecinos forman una secuencia local coherente y separa la captura en
segmentos ante retrocesos o huecos restantes. Nunca crea ventanas a través de esos cortes.

## Pruebas

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\olive\anaconda3\Scripts\conda.exe' run -n ctnet python -m unittest discover -s tests -v
```

## K-Fold de modelos base

Este comando es largo y no debe ejecutarse hasta recibir autorización explícita:

```powershell
& 'C:\Users\olive\anaconda3\envs\ctnet\python.exe' scripts\run_baselines.py `
  --models ctnet,eegconformer,atcnet,msvtnet,lstm --seeds 42 --device cuda
```

El script entrena CTNet, EEGConformer, ATCNet, MSVTNet y PaperLSTM sobre los mismos folds.
Para cada fold separa usuarios internos para early stopping y evalúa sobre usuarios nunca
vistos por ese entrenamiento. El test fijo no se abre.

Los folds se construyen con `GroupKFold`. Se prueban de forma determinista varios estados
de mezcla y se conserva el que, manteniendo 3–4 usuarios por fold y las cuatro clases,
minimiza el desequilibrio de ventanas y proporciones de clase.

Comparación posterior:

```powershell
& 'C:\Users\olive\anaconda3\Scripts\conda.exe' run -n ctnet python scripts\compare_models.py `
  --run-id ID_DE_LA_CORRIDA --primary-metric f1_macro
& 'C:\Users\olive\anaconda3\Scripts\conda.exe' run -n ctnet python scripts\build_report.py `
  --run-id ID_DE_LA_CORRIDA
```

Después de la ronda base se añadieron, sin modificar `PaperLSTM`, las variantes
`AttentionBiLSTM` y `CompactAttentionBiLSTM` en `models/improved_lstm.py`. La segunda,
con 15,776 parámetros, dropout alto, LayerNorm, bidireccionalidad y atención temporal,
fue la ganadora de desarrollo.

La comparación genera intervalos de confianza del 95 % mediante bootstrap agrupado por
usuario y pruebas de Wilcoxon sobre usuarios emparejados, con corrección de Holm.

La evaluación final está protegida por una bandera explícita para evitar abrir el test por
accidente:

```powershell
& 'C:\Users\olive\anaconda3\envs\ctnet\python.exe' scripts\run_final_evaluation.py `
  --models lstm_attention_compact --learning-rate 0.0005 --device cuda `
  --confirm-test-evaluation
```

La evaluación final ya se ejecutó una vez después de congelar el ganador. No debe
repetirse para seguir ajustando el modelo al test.

## Métricas

El evaluador calcula accuracy, balanced accuracy, precision/recall/F1 macro y weighted,
MCC, Cohen's kappa, log-loss, ROC-AUC OVR, PR-AUC, sensibilidad y especificidad por clase.
También produce métricas por usuario, predicciones con probabilidades, matrices de
confusión y curvas de entrenamiento, ROC y Precision-Recall.

Para la tarea binaria también calcula sensibilidad por contexto A/B/C, especificidad en D,
selección de umbral exclusivamente en validación y métricas de eventos (detección, falsas
alarmas por minuto y latencia). El comando principal es:

```powershell
& 'C:\Users\olive\anaconda3\envs\ctnet\python.exe' scripts\run_binary_models.py `
  --models all --seeds 42 --device cuda --run-id ID_DE_LA_CORRIDA
& 'C:\Users\olive\anaconda3\envs\ctnet\python.exe' scripts\compare_binary_models.py `
  --run-id ID_DE_LA_CORRIDA --primary-metric balanced_accuracy
```

Las AUC por usuario se guardan como `NaN` cuando ese usuario no contiene todas las clases;
esto es esperado, especialmente para `S01`.

## Corridas principales completadas

- `20260826_baselines_gpu_seed42`: cinco modelos base, 25 entrenamientos.
- `20260826_lstm_attention_gpu_seed42`: primera variante LSTM.
- `20260826_lstm_attention_compact_gpu_seed42`: variante compacta seleccionada.
- `20260826_all_cv_gpu_seed42`: comparación consolidada, bootstrap y Wilcoxon-Holm.
- `20260826_final_compact_gpu_seed42`: única apertura del test fijo.
- `20260826_binary_all_gpu_seed42`: siete modelos binarios, 35 entrenamientos por usuario.
- `20260901_binary_holdout_exploratory_seed42`: evaluación binaria exploratoria de los
  siete modelos; no debe usarse para volver a seleccionar arquitectura.

El informe multiclase está en `results/final_report.md`. El informe del objetivo clínico
binario está en `results/binary_final_report.md`.

## Entorno

El entorno `ctnet` usa PyTorch `2.9.1+cu128` y la GPU NVIDIA RTX 3050 Ti. CUDA,
entrenamiento con backward, checkpoints, métricas y gráficas fueron validados antes de
la ronda completa.
