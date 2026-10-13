# Evaluación y mitigación de sesgos en reclutamiento

Material experimental del TFM de Jordan Rajo Vázquez, Máster en Inteligencia Artificial. Versión metodológica 2.0.0, octubre de 2026.

**Estudio sintético y controlado. No es una herramienta apta para seleccionar personas.** Las etiquetas representan una rúbrica didáctica, no desempeño laboral observado. Ningún CV pertenece a una persona real. Las desigualdades del entrenamiento se introducen deliberadamente y no son estimaciones del mercado laboral.

## Reproducir

El código se publica como archivos consultables. Los datos, modelos y figuras de la ejecución se distribuyen junto con todo el proyecto en `TFM_Jordan_Rajo_Vazquez_material_completo.zip` (aproximadamente 19 MB). Tras clonar, descomprimir en la raíz con:

```bash
python -m zipfile -e TFM_Jordan_Rajo_Vazquez_material_completo.zip .
```

El archivo también se puede descargar y descomprimir sin Git. `MANIFEST_SHA256.json` permite verificar la integridad del contenido. Las huellas del registro de entrenamiento identifican los scripts presentes al terminar los ajustes; el manifiesto final incluye además las mejoras posteriores de presentación y explicación, sin cambios en los modelos ni en las predicciones.

Python 3.12. En un entorno virtual nuevo:

```bash
git clone https://github.com/jordanrajo-svg/tfm-sesgos-reclutamiento.git
cd tfm-sesgos-reclutamiento
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python reproducir.py
```

`python reproducir.py --quick` ejecuta una comprobación reducida de 200 perfiles y una semilla. Sus resultados **no** son los del TFM. `python reproducir.py --solo-analisis` verifica y vuelve a dibujar los resultados completos guardados. Las figuras de una prueba rápida se escriben en la misma carpeta de figuras: ejecutar de nuevo el análisis completo antes de compilar la memoria.

El ajuste completo utiliza un hilo, 5 corpus independientes, 10.000 perfiles profesionales y 60.000 CVs principales, además de variantes de formato. Son 800 ajustes: 750 para selección de hiperparámetros y 50 de sensibilidad. El tiempo depende del equipo; el tiempo medido y las versiones exactas se guardan en el registro de ejecución.

## Qué contiene

- `src/datos.py`: variables, plantillas, rúbrica y transformaciones textuales.
- `src/experimento.py`: particiones, sesgos, seis tratamientos, LR/SVM, selección por validación y predicciones.
- `src/verificar.py`: comprobación separada de métricas, particiones, etiquetas e hiperparámetros.
- `src/analisis.py`: 13 figuras científicas en PDF/PNG y resúmenes.
- `resultados/datos/`: perfiles, CVs textuales completos, particiones y registros de entrenamiento, con compresión sin pérdida.
- `resultados/predicciones.csv.gz`: predicciones individuales para reconstruir métricas.
- `resultados/modelos/`: seis regresiones logísticas y vectorizadores de la réplica 11, escenario combinado. Cargar archivos joblib únicamente si se confía en su procedencia.
- `resultados/protocolo.json`, `ejecucion.json` y `verificacion.json`: configuración, versiones/huellas de código y controles.
- `figuras/`: visualizaciones generadas a partir de las salidas, nunca valores escritos a mano.
- `src/explicar.py`: reconstrucción exacta de márgenes, con todos los términos y el intercepto.
- `src/memoria.py`: ensamblado de la memoria a partir de las salidas y los componentes de redacción.
- `TFM_Jordan_Rajo_Vazquez_mejorado.tex`: memoria autónoma, con gráficas vectoriales y bibliografía integradas; compilar con pdfLaTeX dos veces. No necesita archivos externos de imágenes o bibliografía.

## Protocolo y limitaciones

Cada corpus tiene 2.000 perfiles (50% positivos por construcción), seis variantes demográficas por perfil y particiones 60/20/20 por perfil. Todas las variantes permanecen juntas. La validación y la prueba usan la rúbrica sin corrupción. El vectorizador aprende solo del entrenamiento. C se selecciona entre 0,1, 1 y 10 por F1 en validación; los empates favorecen el menor C. El umbral principal es cero sobre el margen.

Los cinco escenarios separan representación, corrupción de etiquetas, combinación y control del tamaño muestral. La reponderación equilibra la asociación entre grupo y etiqueta **observada**, no reconstruye automáticamente etiquetas verdaderas. La limpieza completa conoce todas las señales demográficas insertadas por el generador: su invariancia entre variantes es estructural. La auditoría repara las etiquetas de un 30% aleatorio de perfiles de entrenamiento usando la rúbrica conocida; es un supuesto de auditoría perfecta, no una revisión humana efectuada.

El formato no visto cambia encabezados, orden y algunas formulaciones, pero conserva el vocabulario principal; constituye una prueba de cambio de formato moderado, no validación externa. Las categorías de género binarias no representan toda la diversidad. Los intervalos bootstrap remuestrean cinco réplicas completas y pareadas: tienen alcance exploratorio, no incertidumbre sobre una población laboral real.

## Citar

Rajo Vázquez, J. (2026). *Evaluación y mitigación de sesgos en reclutamiento: corpus sintético, protocolo y resultados reproducibles* (versión 2.0.0) [Software y datos]. GitHub. https://github.com/jordanrajo-svg/tfm-sesgos-reclutamiento

Para una entrega académica, añadir el identificador del commit consultado. El repositorio contiene un archivo de cita compatible con GitHub. No se atribuye DOI al material porque no se ha depositado en un archivo que lo asigne.
