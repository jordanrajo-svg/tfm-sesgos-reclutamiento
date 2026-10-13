"""Punto de entrada: ejecutar, verificar y representar el experimento."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--solo-analisis',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent
out='resultados_prueba' if a.quick else 'resultados'
if not a.solo_analisis:subprocess.run([sys.executable,'src/experimento.py','--out',out]+(['--quick'] if a.quick else []),cwd=root,check=True)
for script in ['verificar.py','analisis.py','explicar.py']:subprocess.run([sys.executable,'src/'+script,'--out',out],cwd=root,check=True)
if not a.quick:subprocess.run([sys.executable,'src/memoria.py'],cwd=root,check=True)
