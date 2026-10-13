"""Verificación independiente de predicciones, particiones y selección de C."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,roc_auc_score

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('resultados'));out=ap.parse_args().out
    m=pd.read_csv(out/'metricas.csv');p=pd.read_csv(out/'predicciones.csv.gz');t=pd.read_csv(out/'ajustes.csv');checks=0
    for key,d in p.groupby(['seed','scenario','method','model','domain']):
        seed,sc,met,model,domain=key
        row=m[(m.seed==seed)&(m.scenario==sc)&(m.method==met)&(m.model==model)&(m.domain==domain)].iloc[0]
        assert np.array_equal(d.prediction,(d.score>=0).astype(int))
        assert abs(f1_score(d.y_true,d.prediction)-row.f1)<1e-10
        assert abs(roc_auc_score(d.y_true,d.score)-row.auc)<1e-10
        tpr=d[d.y_true==1].groupby('group').prediction.mean();assert abs(tpr.max()-tpr.min()-row.gap)<1e-10
        assert d.groupby('profile_id').size().eq(6).all()
        tuning=t[(t.seed==seed)&(t.scenario==sc)&(t.method==met)&(t.model==model)].sort_values('C')
        assert row.C==tuning.loc[tuning.validation_f1.idxmax(),'C']
        if met in ['limpieza','limpieza_rw','auditoria_rw']:assert d.groupby('profile_id').score.apply(lambda a:a.max()-a.min()).max()<1e-10
        checks+=7
    for seed in m.seed.unique():
        split=pd.read_csv(out/f'datos/particiones_{seed}.csv');assert split.profile_id.is_unique
        cvs=pd.read_csv(out/f'datos/cvs_{seed}.csv.gz');profiles=pd.read_csv(out/f'datos/perfiles_{seed}.csv.gz')
        q=2*profiles.sql+2*profiles.python+profiles.viz+profiles.stats+(profiles.years>=2)+(profiles.english!='B1');assert np.array_equal((q>=6).astype(int),profiles.y_true)
        assert cvs.groupby('profile_id').y_true.nunique().eq(1).all()
        testids=set(split[split.partition=='test'].profile_id);assert set(p[p.seed==seed].profile_id)==testids
        for sc in m.scenario.unique():
            train=pd.read_csv(out/f'datos/entrenamiento_{seed}_{sc}.csv.gz');assert set(train.profile_id)<=set(split[split.partition=='train'].profile_id)
            assert ((train.y_observed<=train.y_true)).all()
            assert (train.loc[train.audited,'y_audited']==train.loc[train.audited,'y_true']).all()
            assert (train.loc[~train.audited,'y_audited']==train.loc[~train.audited,'y_observed']).all()
            checks+=4
        checks+=4
    report=dict(status='PASS',checks=checks,metric_rows=len(m),prediction_rows=len(p),independent_corpora=len(m.seed.unique()))
    (out/'verificacion.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)

if __name__=='__main__':main()
