"""Figuras y tablas derivadas exclusivamente de resultados guardados."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from datos import GROUPS

LABELS={'original':'Original','reponderacion':'Reponderación','cabecera':'Sin cabecera','limpieza':'Limpieza completa','limpieza_rw':'Limpieza + pesos','auditoria_rw':'Limpieza + pesos + auditoría'}
COLORS=['#a33d3d','#d18d32','#9874a6','#437fa0','#4a987d','#173f66']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('resultados'));args=ap.parse_args();out=args.out
    figdir=out.parent/'figuras';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    def save(name):
        plt.tight_layout();plt.savefig(figdir/(name+'.pdf'));plt.savefig(figdir/(name+'.png'),dpi=170);plt.close()
    m=pd.read_csv(out/'metricas.csv');gr=pd.read_csv(out/'grupos.csv');tr=pd.read_csv(out/'entrenamiento.csv');se=pd.read_csv(out/'sensibilidad.csv')
    base=m[(m.domain=='iid')&(m.scenario=='combinado')]
    agg=m.groupby(['scenario','method','model','domain'])[['f1','gap','auc','gender_flip','age_flip']].agg(['mean','std']);agg.to_csv(out/'resumen.csv')
    fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for ax,metric,title in zip(axes,['f1','gap'],['F1 (mayor es mejor)','Brecha TPR (menor es mejor)']):
        for j,model in enumerate(['LR','SVM']):
            z=base[base.model==model].groupby('method')[metric].agg(['mean','std']).reindex(LABELS)
            ax.barh(np.arange(6)+j*.34,z['mean'],xerr=z['std'],height=.32,label=model,color=['#27648a','#c17b31'][j],capsize=2)
        ax.set_yticks(np.arange(6)+.17,list(LABELS.values()) if ax==axes[0] else []);ax.invert_yaxis();ax.set_xlabel(title);ax.set_xlim(0,1);ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=2)
    save('01_comparacion')
    fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for ax,metric in zip(axes,['f1','gap']):
        z=m[(m.domain=='iid')&(m.method=='original')].groupby(['scenario','model'])[metric].mean().unstack().reindex(['control','control_n','representacion','etiquetado','combinado'])
        z.plot.bar(ax=ax,color=['#27648a','#c17b31'],rot=25);ax.set_ylabel(metric.upper() if metric=='f1' else 'Brecha TPR');ax.set_xlabel('');ax.set_ylim(0,1)
    save('02_escenarios')
    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    for ax,model in zip(axes,['LR','SVM']):
        z=gr[(gr.domain=='iid')&(gr.scenario=='combinado')&(gr.model==model)].groupby(['method','group']).tpr.mean().unstack().reindex(LABELS)
        im=ax.imshow(z,vmin=0,vmax=1,cmap='Blues',aspect='auto');ax.set_xticks(range(6),GROUPS,rotation=45,ha='right');ax.set_yticks(range(6),list(LABELS.values()) if ax==axes[0] else []);ax.set_title(model)
        for i in range(6):
            for j in range(6):ax.text(j,i,f'{z.iloc[i,j]:.2f}',ha='center',va='center',color='white' if z.iloc[i,j]>.65 else 'black',fontsize=8)
    fig.suptitle('TPR por grupo y tratamiento',y=1.04);plt.tight_layout();plt.savefig(figdir/'03_grupos.pdf',bbox_inches='tight');plt.savefig(figdir/'03_grupos.png',dpi=170,bbox_inches='tight');plt.close()
    fig,ax=plt.subplots(figsize=(7,3.5))
    for color,(method,label) in zip(COLORS,LABELS.items()):
        z=base[(base.method==method)&(base.model=='LR')];ax.errorbar(z.gap.mean(),z.f1.mean(),xerr=z.gap.std(),yerr=z.f1.std(),fmt='o',color=color,label=label,capsize=3)
    ax.set_xlabel('Brecha TPR (menor es mejor)');ax.set_ylabel('F1 (mayor es mejor)');ax.legend(loc='lower left',fontsize=8);save('04_compromiso')
    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    z=tr[tr.scenario=='combinado'].groupby('group').mean(numeric_only=True)
    axes[0].bar(GROUPS,z.n,color='#27648a');axes[0].set_ylabel('CVs retenidos (media)')
    axes[1].bar(GROUPS,1-z.positive_observed/z.positive_true,color='#a33d3d');axes[1].set_ylabel('Fracción de positivos corrompidos');axes[1].set_ylim(0,1)
    for ax in axes:ax.tick_params(axis='x',rotation=40)
    save('05_sesgos_datos')
    fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for ax,model in zip(axes,['LR','SVM']):
        z=m[(m.scenario=='combinado')&(m.model==model)].groupby(['method','domain']).f1.mean().unstack().reindex(LABELS)
        ax.plot(range(6),z.iid,'o-',label='Formato de desarrollo');ax.plot(range(6),z.ood,'s--',label='Formato no visto');ax.set_xticks(range(6),list(LABELS.values()),rotation=40,ha='right');ax.set_ylim(0,1);ax.set_ylabel('F1');ax.set_title(model);ax.legend(fontsize=7)
    save('06_formatos')
    stored=json.loads((out/'predicciones_representativas.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    for color,d in zip(COLORS,stored):
        if d['method'] not in ['original','cabecera','auditoria_rw']:continue
        fpr,tpr,_=roc_curve(d['y'],d['score']);pre,rec,_=precision_recall_curve(d['y'],d['score']);axes[0].plot(fpr,tpr,color=color,label=LABELS[d['method']]);axes[1].plot(rec,pre,color=color,label=LABELS[d['method']])
    axes[0].plot([0,1],[0,1],':',color='gray');axes[0].set(xlabel='FPR',ylabel='TPR');axes[1].set(xlabel='Recall',ylabel='Precisión');axes[1].legend(fontsize=7);save('07_curvas')
    u=pd.read_csv(out/'umbrales_validacion.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    for method in ['original','auditoria_rw']:
        z=u[u.method==method].groupby('threshold')[['f1','gap']].mean()
        for ax,metric in zip(axes,['f1','gap']):ax.plot(z.index,z[metric],label=LABELS[method]);ax.set_xlabel('Umbral sobre margen');ax.set_ylabel('F1' if metric=='f1' else 'Brecha TPR');ax.axvline(0,color='gray',ls=':')
    axes[0].legend(fontsize=7);save('08_umbrales')
    fig,axes=plt.subplots(1,2,figsize=(8,3.2));z=se[se.analysis=='corrupcion'].copy();z['setting']=z.setting.astype(float)
    for ax,metric in zip(axes,['f1','gap']):
        a=z.groupby('setting')[metric].agg(['mean','std']);ax.errorbar(a.index,a['mean'],yerr=a['std'],fmt='o-',capsize=3);ax.set_xlabel('Multiplicador de corrupción');ax.set_ylabel(metric.upper() if metric=='f1' else 'Brecha TPR');ax.set_ylim(0,1)
    save('09_intensidad')
    fig,ax=plt.subplots(figsize=(7,3.1));z=se[se.analysis=='rubrica'].copy();z['rubric']=z.setting.str.split('_').str[0];z['method']=z.setting.str.replace(r'^[^_]+_','',regex=True)
    z.groupby(['rubric','method']).f1.mean().unstack().rename(columns=LABELS).plot.bar(ax=ax,color=COLORS[-2:],rot=0);ax.set_ylabel('F1');ax.set_xlabel('Rúbrica alternativa');ax.set_ylim(0,1);ax.legend(fontsize=8);save('10_rubricas')
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    profiles=pd.read_csv(out/'datos/perfiles_11.csv.gz');cvs=pd.read_csv(out/'datos/cvs_11.csv.gz')
    axes[0].hist(cvs.age,bins=np.arange(23.5,63.5,2),color='#27648a');axes[0].set(xlabel='Edad sintética',ylabel='CVs (semilla 11)')
    axes[1].hist(cvs.iloc[::6].text.str.split().str.len(),bins=18,color='#4a987d');axes[1].set(xlabel='Palabras por CV',ylabel='Perfiles (semilla 11)');save('11_diversidad')
    e=pd.read_csv(out/'explicaciones.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.4))
    for ax,method in zip(axes,['original','auditoria_rw']):
        z=e[(e.kind=='global')&(e.method==method)].sort_values('value',key=abs).tail(10).sort_values('value');ax.barh(z.term,z.value,color=['#a33d3d' if v<0 else '#27648a' for v in z.value]);ax.set_title(LABELS[method]);ax.set_xlabel('Coeficiente (no efecto causal)')
    save('12_coeficientes')
    # Intervalos pareados entre réplicas independientes: bootstrap de las 5 medias.
    # Unidad de remuestreo: corpus + ajuste. No tratar los CVs hermanos como independientes.
    rng=np.random.default_rng(20261013);comparisons=[]
    for model in ['LR','SVM']:
        for method in ['cabecera','limpieza_rw','auditoria_rw']:
            for metric in ['f1','gap']:
                a=base[(base.model==model)&(base.method==method)].set_index('seed')[metric];b=base[(base.model==model)&(base.method=='original')].set_index('seed')[metric];d=(a-b).sort_index().to_numpy();boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(axis=1)
                comparisons.append(dict(model=model,method=method,metric=metric,delta=float(d.mean()),low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),n_replicas=len(d)))
    pd.DataFrame(comparisons).to_csv(out/'intervalos_pareados.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8,3.3))
    for ax,metric in zip(axes,['f1','gap']):
        z=pd.DataFrame(comparisons);z=z[(z.metric==metric)&(z.model=='LR')];ax.errorbar(z.delta,range(3),xerr=[z.delta-z.low,z.high-z.delta],fmt='o',capsize=4);ax.set_yticks(range(3),[LABELS[v] for v in z.method] if metric=='f1' else []);ax.axvline(0,color='gray',ls=':');ax.set_xlabel('Cambio de F1' if metric=='f1' else 'Cambio de brecha TPR')
    save('13_incertidumbre')
    # Resumen compacto para incorporar números sin transcribirlos manualmente.
    summary={}
    for model in ['LR','SVM']:
        summary[model]={}
        for method in LABELS:
            z=base[(base.model==model)&(base.method==method)]
            summary[model][method]={k:dict(mean=float(z[k].mean()),sd=float(z[k].std())) for k in ['f1','gap','auc','gender_flip','age_flip']}
    (out/'sintesis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
