"""Reconstruye márgenes de LR sin aproximación ni explicador externo.

Solo cargar modelos joblib de procedencia conocida. Los modelos del paquete
corresponden al escenario combinado, réplica 11.
"""
import argparse,csv,json
from pathlib import Path
import joblib
import numpy as np
from datos import clean_text,suppress_header

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('resultados'));out=ap.parse_args().out
    examples=json.loads((out/'ejemplos_cvs.json').read_text(encoding='utf-8'));terms=[];summary=[]
    for path in sorted((out/'modelos').glob('*.joblib')):
        bundle=joblib.load(path);v=bundle['vectorizer'];model=bundle['model'];representation=bundle['representation']
        for cv in examples:
            text=cv['text'];text=clean_text(text) if representation=='clean' else suppress_header(text) if representation=='header' else text
            x=v.transform([text]);contribution=x.multiply(model.coef_[0]).tocsr();intercept=float(model.intercept_[0]);total=float(contribution.sum())+intercept;score=float(model.decision_function(x)[0]);assert abs(total-score)<1e-10
            names=v.get_feature_names_out()
            for j,value in zip(contribution.indices,contribution.data):terms.append(dict(method=path.stem,cv_id=cv['cv_id'],term=names[j],tfidf=float(x[0,j]),coefficient=float(model.coef_[0,j]),contribution=float(value)))
            summary.append(dict(method=path.stem,cv_id=cv['cv_id'],profile_id=cv['profile_id'],group=cv['group'],y_true=cv['y_true'],intercept=intercept,sum_contributions=float(contribution.sum()),score=score,prediction=int(score>=0)))
    for name,rows in [('contribuciones_completas.csv',terms),('explicacion_casos.csv',summary)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(f'{len(summary)} decisiones reconstruidas exactamente; {len(terms)} contribuciones léxicas.')

if __name__=='__main__':main()
