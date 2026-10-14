"""Ejecuta el experimento, verifica métricas y genera figuras y explicaciones.

Autor: Jordan Rajo Vázquez (2026). Python 3.12.
Estudio controlado con CVs sintéticos; no evalúa personas reales.

Instalación en un entorno virtual:
    python -m pip install numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 joblib==1.6.0 threadpoolctl==3.7.0 pandas==3.0.1 matplotlib==3.11.2

Ejecución completa: python reproducir.py
Prueba reducida: python reproducir.py --quick
Reanalizar resultados existentes: python reproducir.py --solo-analisis

Las salidas se crean localmente en resultados/ (o resultados_prueba/) y
figuras/. La prueba reducida no reproduce los resultados completos del TFM
y utiliza la misma carpeta figuras/. No se genera ningún documento LaTeX.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='Prueba reducida de 200 perfiles y una semilla.')
    parser.add_argument('--solo-analisis', action='store_true', help='Utilizar resultados ya generados.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = 'resultados_prueba' if args.quick else 'resultados'
    if not args.solo_analisis:
        subprocess.run([sys.executable, 'src/experimento.py', '--out', out] + (['--quick'] if args.quick else []), cwd=root, check=True)
    for script in ['verificar.py', 'analisis.py', 'explicar.py']:
        subprocess.run([sys.executable, 'src/' + script, '--out', out], cwd=root, check=True)


if __name__ == '__main__':
    main()
