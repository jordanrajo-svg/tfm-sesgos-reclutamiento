"""Experimento V2: protocolo congelado antes de ajustar modelos.
Ejecutar desde la raíz: python src/experimento.py
No usa APIs, CVs reales ni modelos de lenguaje externos.
"""
import argparse,csv,gzip,hashlib,json,platform,time,warnings
from pathlib import Path
import numpy as np
import sklearn,scipy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,roc_auc_score,average_precision_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from datos import generate,clean_text,suppress_header,rubric

CONFIG=dict(version='2.0.0',n_profiles=2000,seeds=[11,22,33,44,55],generation_seed_offset=20261013,C=[.1,1,10],retention=[1,.90,.65,.70,.50,.25],corruption=[.05,.15,.40,.20,.35,.60],
 scenarios=['control','representacion','etiquetado','combinado','control_n'],methods=['original','reponderacion','cabecera','limpieza','limpieza_rw','auditoria_rw'],
 audit_fraction=.30,tfidf=dict(ngram_range=[1,2],min_df=2,max_features=14000,sublinear_tf=True,strip_accents='unicode',token_pattern=r'(?u)\b\w+\b'),max_iter=10000,tol=1e-5)

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def savecsv(path,rows):
    if not rows:return
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def reweight(g,y):
    n=len(y);w=np.ones(n)
    for a in range(6):
        for b in [0,1]:
            mask=(g==a)&(y==b)
            if not mask.any():raise ValueError('Celda vacía en reponderación')
            w[mask]=(g==a).sum()*(y==b).sum()/(n*mask.sum())
    assert abs(w.mean()-1)<1e-10
    return w
def rates(y,p,g):
    out=[]
    for a in range(6):
        m=g==a;pos=m&(y==1);neg=m&(y==0)
        out.append(dict(group=a,n=int(m.sum()),tp=int(p[pos].sum()),fn=int((1-p[pos]).sum()),fp=int(p[neg].sum()),tn=int((1-p[neg]).sum()),tpr=float(p[pos].mean()),fpr=float(p[neg].mean()),sr=float(p[m].mean())))
    return out
def metrics(y,p,s,g):
    rr=rates(y,p,g);tp=[r['tpr'] for r in rr];fp=[r['fpr'] for r in rr]
    pp=p.reshape(-1,6)
    return dict(f1=float(f1_score(y,p)),accuracy=float(accuracy_score(y,p)),precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p)),auc=float(roc_auc_score(y,s)),ap=float(average_precision_score(y,s)),gap=float(np.ptp(tp)),eodds=float(max(np.ptp(tp),np.ptp(fp))),dp=float(np.ptp([r['sr'] for r in rr])),gender_flip=float(np.mean(pp[:,:3]!=pp[:,3:])),age_flip=float(np.mean([pp[:,i]!=pp[:,j] for i,j in [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]])))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('resultados'));parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    if args.quick:CONFIG.update(n_profiles=200,seeds=[11],C=[1])
    out=args.out;out.mkdir(parents=True,exist_ok=True);(out/'datos').mkdir(exist_ok=True)
    dump(out/'protocolo.json',CONFIG);start=time.perf_counter()
    allmetrics=[];allgroups=[];tuning=[];training=[];thresholds=[];sensitivity=[];explanations=[];stored=[];fitcount=0
    with gzip.open(out/'predicciones.csv.gz','wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','scenario','method','model','domain','cv_id','profile_id','group','y_true','score','prediction']);writer.writeheader()
        for seed in CONFIG['seeds']:
            profiles,rows=generate(CONFIG['n_profiles'],CONFIG['generation_seed_offset']+seed)
            savecsv(out/'datos'/f'perfiles_{seed}.csv.gz',profiles);savecsv(out/'datos'/f'cvs_{seed}.csv.gz',rows)
            ybase=np.array([p['y_true'] for p in profiles]);y=np.array([r['y_true'] for r in rows]);pid=np.array([r['profile_id'] for r in rows]);g=np.array([r['group'] for r in rows])
            tr,rest=train_test_split(np.arange(len(profiles)),test_size=.4,stratify=ybase,random_state=seed);va,te=train_test_split(rest,test_size=.5,stratify=ybase[rest],random_state=seed)
            assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
            savecsv(out/'datos'/f'particiones_{seed}.csv',[dict(profile_id=int(i),partition=part) for part,ids in [('train',tr),('validation',va),('test',te)] for i in ids])
            trall=np.flatnonzero(np.isin(pid,tr));val=np.flatnonzero(np.isin(pid,va));test=np.flatnonzero(np.isin(pid,te))
            rng=np.random.default_rng(seed+80000);retain=rng.random(len(rows));flip=rng.random(len(rows));audit_ids=rng.choice(tr,size=round(len(tr)*CONFIG['audit_fraction']),replace=False)
            kept=trall[retain[trall]<np.array(CONFIG['retention'])[g[trall]]]
            # Control uniforme del mismo tamaño total: estratificación por grupo y clase.
            uniform=[];cells=[trall[(g[trall]==a)&(y[trall]==b)] for a in range(6) for b in [0,1]]
            for k,cell in enumerate(cells):uniform.extend(rng.choice(cell,len(kept)//12+(k<len(kept)%12),replace=False))
            uniform=np.sort(uniform)
            texts={}
            for domain,key in [('iid','text'),('ood','text_ood')]:
                raw=[r[key] for r in rows];texts[domain]={'raw':raw,'header':[suppress_header(s) for s in raw],'clean':[clean_text(s) for s in raw]}
            for scenario in CONFIG['scenarios']:
                train=uniform if scenario=='control_n' else kept if scenario in ['representacion','combinado'] else trall
                yt=y[train].copy()
                if scenario in ['etiquetado','combinado']:yt[(yt==1)&(flip[train]<np.array(CONFIG['corruption'])[g[train]])]=0
                audit=np.isin(pid[train],audit_ids);ya=yt.copy();ya[audit]=y[train][audit]
                savecsv(out/'datos'/f'entrenamiento_{seed}_{scenario}.csv.gz',[dict(cv_id=int(i),profile_id=int(pid[i]),group=int(g[i]),y_true=int(y[i]),y_observed=int(a),audited=bool(b),y_audited=int(c)) for i,a,b,c in zip(train,yt,audit,ya)])
                for a in range(6):
                    m=g[train]==a;training.append(dict(seed=seed,scenario=scenario,group=a,n=int(m.sum()),positive_true=int(y[train][m].sum()),positive_observed=int(yt[m].sum()),audit_n=int(audit[m].sum()),repaired=int(np.sum(yt[m]!=ya[m]))))
                methods=CONFIG['methods'] if scenario!='control_n' else ['original']
                cache={}
                for method in methods:
                    representation='raw' if method in ['original','reponderacion'] else 'header' if method=='cabecera' else 'clean'
                    if representation not in cache:
                        pars=dict(CONFIG['tfidf']);pars['ngram_range']=tuple(pars['ngram_range']);vect=TfidfVectorizer(**pars)
                        xtr=vect.fit_transform([texts['iid'][representation][i] for i in train]);xva=vect.transform([texts['iid'][representation][i] for i in val]);xt=vect.transform([texts['iid'][representation][i] for i in test]);xo=vect.transform([texts['ood'][representation][i] for i in test]);cache[representation]=(vect,xtr,xva,xt,xo)
                    vect,xtr,xva,xt,xo=cache[representation];target=ya if method=='auditoria_rw' else yt
                    weights=reweight(g[train],target) if method in ['reponderacion','limpieza_rw','auditoria_rw'] else None
                    for modelname in ['LR','SVM']:
                        best=None
                        for C in CONFIG['C']:
                            model=LogisticRegression(C=C,solver='liblinear',max_iter=CONFIG['max_iter'],tol=CONFIG['tol'],random_state=seed) if modelname=='LR' else LinearSVC(C=C,dual=False,max_iter=CONFIG['max_iter'],tol=CONFIG['tol'],random_state=seed)
                            with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);model.fit(xtr,target,sample_weight=weights)
                            fitcount+=1;vf=float(f1_score(y[val],model.decision_function(xva)>=0));tuning.append(dict(seed=seed,scenario=scenario,method=method,model=modelname,C=C,validation_f1=vf))
                            if best is None or vf>best[0]+1e-12:best=(vf,C,model)
                        vf,C,model=best
                        if seed==11 and scenario=='combinado' and modelname=='LR':
                            (out/'modelos').mkdir(exist_ok=True)
                            joblib.dump(dict(vectorizer=vect,model=model,representation=representation),out/'modelos'/f'{method}.joblib',compress=3)
                        for domain,X in [('iid',xt),('ood',xo)]:
                            score=model.decision_function(X);pred=(score>=0).astype(int);meta=dict(seed=seed,scenario=scenario,method=method,model=modelname,domain=domain)
                            allmetrics.append(dict(**meta,C=C,n_train=len(train),features=xtr.shape[1],validation_f1=vf,**metrics(y[test],pred,score,g[test])))
                            allgroups.extend(dict(**meta,**r) for r in rates(y[test],pred,g[test]))
                            writer.writerows(dict(**meta,cv_id=int(i),profile_id=int(pid[i]),group=int(g[i]),y_true=int(y[i]),score=float(s),prediction=int(p)) for i,s,p in zip(test,score,pred))
                            if seed==11 and scenario=='combinado' and modelname=='LR' and domain=='iid':
                                stored.append(dict(method=method,test=test.tolist(),profile_ids=pid[test].tolist(),y=y[test].tolist(),g=g[test].tolist(),score=score.tolist(),pred=pred.tolist()))
                                if method in ['original','cabecera','auditoria_rw']:
                                    names=vect.get_feature_names_out();coef=model.coef_[0]
                                    for i in np.argsort(np.abs(coef))[-20:][::-1]:explanations.append(dict(method=method,kind='global',cv_id=-1,term=str(names[i]),value=float(coef[i]),intercept=float(model.intercept_[0])))
                                    # Mismo perfil positivo en todos los tratamientos, seleccionado por ID.
                                    ip=int(np.flatnonzero(y[test]==1)[0]//6*6)
                                    for rowidx in [ip,ip+5]:
                                        contrib=X[rowidx].multiply(coef).toarray()[0]
                                        for i in np.argsort(np.abs(contrib))[-12:][::-1]:explanations.append(dict(method=method,kind='local',cv_id=int(test[rowidx]),term=str(names[i]),value=float(contrib[i]),intercept=float(model.intercept_[0])))
                        if scenario=='combinado' and modelname=='LR' and method in ['original','auditoria_rw']:
                            sv=model.decision_function(xva);st=model.decision_function(xt)
                            grid=np.unique(np.r_[np.linspace(-3,3,61),0]);vals=[]
                            for threshold in grid:
                                mv=metrics(y[val],(sv>=threshold).astype(int),sv,g[val]);vals.append(mv)
                                thresholds.append(dict(seed=seed,method=method,threshold=float(threshold),**mv))
                            # Selección solo en validación; no busca umbrales distintos por grupo.
                            pick=max(range(len(grid)),key=lambda j:(vals[j]['f1'],-abs(grid[j])))
                            sensitivity.append(dict(seed=seed,analysis='umbral',setting=method,threshold=float(grid[pick]),**metrics(y[test],(st>=grid[pick]).astype(int),st,g[test])))
                print(f'seed={seed} scenario={scenario} complete; fits={fitcount}',flush=True)
            # Sensibilidad controlada: LR, solo escenario combinado, C=1 predefinido.
            vect,xtr,xva,xt,xo=cache['raw'] if 'raw' in cache else (None,)*5
            # Reajusta representación sobre el combinado, no sobre control_n.
            pars=dict(CONFIG['tfidf']);pars['ngram_range']=tuple(pars['ngram_range']);vv=TfidfVectorizer(**pars)
            xx=vv.fit_transform([texts['iid']['raw'][i] for i in kept]);xxte=vv.transform([texts['iid']['raw'][i] for i in test])
            for intensity in [0,.5,1,1.5]:
                yy=y[kept].copy();yy[(yy==1)&(flip[kept]<np.minimum(.95,np.array(CONFIG['corruption'])[g[kept]]*intensity))]=0
                mm=LogisticRegression(C=1,solver='liblinear',max_iter=10000,tol=1e-5).fit(xx,yy);ss=mm.decision_function(xxte);fitcount+=1
                sensitivity.append(dict(seed=seed,analysis='corrupcion',setting=str(intensity),threshold=0.,**metrics(y[test],(ss>=0).astype(int),ss,g[test])))
            # Reentrena para cada rúbrica alternativa; no reinterpreta predicciones antiguas.
            cc=TfidfVectorizer(**pars);xc=cc.fit_transform([texts['iid']['clean'][i] for i in kept]);xcv=cc.transform([texts['iid']['clean'][i] for i in test])
            for variant in ['laxa','estricta','igual']:
                yr=np.array([rubric(profiles[int(p)],variant) for p in pid]);obs=yr[kept].copy();obs[(obs==1)&(flip[kept]<np.array(CONFIG['corruption'])[g[kept]])]=0
                for met in ['limpieza_rw','auditoria_rw']:
                    target=obs.copy()
                    if met=='auditoria_rw':am=np.isin(pid[kept],audit_ids);target[am]=yr[kept][am]
                    mm=LogisticRegression(C=1,solver='liblinear',max_iter=10000,tol=1e-5).fit(xc,target,sample_weight=reweight(g[kept],target));ss=mm.decision_function(xcv);fitcount+=1
                    sensitivity.append(dict(seed=seed,analysis='rubrica',setting=variant+'_'+met,threshold=0.,**metrics(yr[test],(ss>=0).astype(int),ss,g[test])))
            if seed==11:
                dump(out/'predicciones_representativas.json',stored)
                exampleids=sorted(set(e['cv_id'] for e in explanations if e['cv_id']>=0))
                exampleids+= [int(test[np.flatnonzero(y[test]==0)[0]])]
                dump(out/'ejemplos_cvs.json',[dict(**rows[i],reference_score=profiles[rows[i]['profile_id']]['score']) for i in sorted(set(exampleids))])
    for name,data in [('metricas',allmetrics),('grupos',allgroups),('ajustes',tuning),('entrenamiento',training),('umbrales_validacion',thresholds),('sensibilidad',sensitivity),('explicaciones',explanations)]:savecsv(out/(name+'.csv'),data)
    import importlib.metadata as im
    dump(out/'ejecucion.json',dict(date=time.strftime('%Y-%m-%d'),seconds=time.perf_counter()-start,fits=fitcount,selected_models=len(allmetrics)//2,python=platform.python_version(),platform=platform.platform(),versions={k:im.version(k) for k in ['numpy','scipy','scikit-learn','matplotlib','pandas','threadpoolctl']},source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    print('FINISHED',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
