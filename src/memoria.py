"""Ensambla la memoria autónoma con valores y gráficos derivados de CSV."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'resultados'
M=pd.read_csv(R/'metricas.csv');G=pd.read_csv(R/'grupos.csv');S=pd.read_csv(R/'sensibilidad.csv')
METHODS=['original','reponderacion','cabecera','limpieza','limpieza_rw','auditoria_rw']
NAMES=['Original','Reponderación','Sin cabecera','Limpieza','Limpieza + pesos','Limpieza + pesos + auditoría']
COLORS=['red!65!black','orange!80!black','violet','cyan!60!black','green!45!black','blue!65!black']
def num(v):return f'{v:.3f}'.replace('.',',')
def esc(s):
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=str(s).replace(a,b)
    return s
def mean(method,metric='f1',model='LR',scenario='combinado',domain='iid'):
    return M[(M.method==method)&(M.model==model)&(M.scenario==scenario)&(M.domain==domain)][metric].mean()
def coords(x,y):return ' '.join(f'({float(a):.5f},{float(b):.5f})' for a,b in zip(x,y))
def axis(series,xlabels=None,ylabel='',xlabel='',width=r'.46\textwidth',height='4.1cm',extra=''):
    opts=f'width={width},height={height},scale only axis=false,ylabel={{{ylabel}}},xlabel={{{xlabel}}},tick label style={{font=\\scriptsize}},label style={{font=\\small}},legend style={{font=\\tiny,at={{(0.5,-0.27)}},anchor=north,legend columns=2}},grid=major,grid style={{gray!15}},{extra}'
    if xlabels:opts+=',xtick={'+','.join(str(i) for i in range(len(xlabels)))+'},xticklabels={'+','.join(xlabels)+'}'
    s=r'\begin{tikzpicture}\begin{axis}['+opts+']\n'
    for j,(x,y,label) in enumerate(series):
        s+=r'\addplot+[thick,mark=*,mark size=1.4pt,color='+COLORS[j]+'] coordinates {'+coords(x,y)+'};\n'
        if label:s+=r'\addlegendentry{'+esc(label)+'}\n'
    return s+r'\end{axis}\end{tikzpicture}'
def fig(content,caption,label):return '\n'+r'\begin{figure}[htbp]\centering'+'\n'+content+'\n'+r'\caption['+caption.split('.')[0]+']{'+caption+r'}\label{fig:'+label+r'}\end{figure}'+'\n'
def diagram(nodes,caption,label):
    body=r'\begin{tikzpicture}[node distance=3mm, every node/.style={draw=blue!40!black,rounded corners,fill=blue!4,text width=11.9cm,align=center,font=\small,inner sep=5pt}]'+'\n'
    for i,t in enumerate(nodes):
        body+=r'\node'+('' if i==0 else f'[below=of n{i-1}]')+f' (n{i}) '+'{'+t+'};\n'
        if i:body+=f'\\draw[-{{Latex}},thick] (n{i-1}) -- (n{i});\n'
    return fig(body+r'\end{tikzpicture}',caption,label)

def figures():
    out={}
    a=[]
    for metric in ['f1','gap']:
        a.append(axis([(range(6),[mean(k,metric,model) for k in METHODS],model) for model in ['LR','SVM']],['M0','M1','M2','M3','M4','M5'],ylabel='F1' if metric=='f1' else r'$\Delta_{TPR}$',extra='ymin=0,ymax=1'))
    out['comparacion']=fig(''.join(a),'Comparación principal en el escenario combinado. Medias de cinco corpus. M0: original; M1: reponderación; M2: sin cabecera; M3: limpieza; M4: limpieza y pesos; M5: limpieza, pesos y auditoría. Las líneas unen categorías solo como ayuda visual.','comparacion')
    scenes=['control','control_n','representacion','etiquetado','combinado'];a=[]
    for metric in ['f1','gap']:a.append(axis([(range(5),[mean('original',metric,model,sc) for sc in scenes],model) for model in ['LR','SVM']],['C','N','R','E','RE'],ylabel='F1' if metric=='f1' else r'$\Delta_{TPR}$',extra='ymin=0,ymax=1'))
    out['escenarios']=fig(''.join(a),'Modelos originales. C: control; N: control de tamaño; R: representación; E: etiquetado; RE: combinado. Cada punto promedia cinco réplicas.','escenarios')
    a=[]
    for model in ['LR','SVM']:
        d=G[(G.model==model)&(G.domain=='iid')&(G.scenario=='combinado')].groupby(['method','group']).tpr.mean().unstack().reindex(METHODS)
        text=r'\begin{tikzpicture}[x=.58cm,y=.44cm]'+'\n'
        for i in range(6):
            for j in range(6):
                v=d.iloc[i,j];text+=f'\\fill[blue!{int(v*80)}!white] ({j},{-i}) rectangle ({j+1},{-i-1});\n'+f'\\node[font=\\tiny,text='+('white' if v>.65 else 'black')+f'] at ({j+.5},{-i-.5}) '+'{'+f'{v:.2f}'+'};\n'
            text+=f'\\node[anchor=east,font=\\scriptsize] at (0,{-i-.5}) '+'{M'+str(i)+'};\n'
        for j,l in enumerate(['H1','H2','H3','M1','M2','M3']):text+=f'\\node[font=\\scriptsize] at ({j+.5},.4) '+'{'+l+'};\n'
        text+=r'\node at (3,1.2) {'+model+r'};\end{tikzpicture}'
        a.append(text)
    out['grupos']=fig(r'\hspace{4mm}'.join(a),'Sensibilidad (TPR) por tratamiento e intersección en el escenario combinado. H/M: género; 1, 2 y 3: intervalos de edad de menor a mayor. Filas M0--M5 como en la figura anterior. Medias de cinco corpus.','grupos')
    out['compromiso']=fig(axis([([mean(k,'gap')],[mean(k)],'M'+str(i)) for i,k in enumerate(METHODS)],ylabel='F1',xlabel=r'$\Delta_{TPR}$',width=r'.8\textwidth',height='4.5cm',extra='only marks,ymin=0,ymax=1'),'Rendimiento y brecha de oportunidad de LR, escenario combinado. Los puntos pueden solaparse cuando la brecha es cero; las cifras exactas se incluyen en la tabla principal.','compromiso')
    a=[];t=pd.read_csv(R/'entrenamiento.csv');z=t[t.scenario=='combinado'].groupby('group').mean(numeric_only=True)
    a.append(axis([(range(6),z.n,'')],['H1','H2','H3','M1','M2','M3'],ylabel='CVs retenidos',extra='ybar,ymin=0'))
    a.append(axis([(range(6),1-z.positive_observed/z.positive_true,'')],['H1','H2','H3','M1','M2','M3'],ylabel='Positivos corrompidos',extra='ybar,ymin=0,ymax=1'))
    out['datos']=fig(''.join(a),'Composición efectiva del entrenamiento combinado: número retenido y proporción de positivos convertidos en negativos. Promedios sobre cinco corpus.','datos')
    a=[]
    for model in ['LR','SVM']:a.append(axis([(range(6),[mean(k,'f1',model,domain=d) for k in METHODS],model+' '+lab) for d,lab in [('iid','habitual'),('ood','nuevo')]],['M0','M1','M2','M3','M4','M5'],ylabel='F1',extra='ymin=0,ymax=1'))
    out['formatos']=fig(''.join(a),'Prueba de formato reservado. Se evalúan los mismos perfiles y modelos; solo cambia la presentación del texto. No se reajusta el clasificador.','formatos')
    from sklearn.metrics import roc_curve,precision_recall_curve
    ds=json.loads((R/'predicciones_representativas.json').read_text(encoding='utf-8'));roc=[];pr=[]
    for d in ds:
        if d['method'] not in ['original','cabecera','auditoria_rw']:continue
        x,y,_=roc_curve(d['y'],d['score']);ix=np.unique(np.linspace(0,len(x)-1,min(90,len(x))).astype(int));roc.append((x[ix],y[ix],'M'+str(METHODS.index(d['method']))))
        y,x,_=precision_recall_curve(d['y'],d['score']);ix=np.unique(np.linspace(0,len(x)-1,min(90,len(x))).astype(int));pr.append((x[ix],y[ix],'M'+str(METHODS.index(d['method']))))
    out['curvas']=fig(axis(roc,xlabel='FPR',ylabel='TPR',extra='xmin=0,xmax=1,ymin=0,ymax=1,mark=none')+axis(pr,xlabel='Recall',ylabel='Precisión',extra='xmin=0,xmax=1,ymin=0,ymax=1,mark=none'),'Curvas ROC y precisión--recall de LR, semilla 11, escenario combinado. Réplica ilustrativa, no agregación de cinco ejecuciones. Se muestran hasta 90 puntos por curva para aligerar el documento.','curvas')
    u=pd.read_csv(R/'umbrales_validacion.csv');a=[]
    for metric in ['f1','gap']:
        series=[]
        for k in ['original','auditoria_rw']:
            z=u[u.method==k].groupby('threshold')[metric].mean();series.append((z.index,z.values,'M'+str(METHODS.index(k))))
        a.append(axis(series,xlabel='Umbral',ylabel='F1' if metric=='f1' else r'$\Delta_{TPR}$',extra='ymin=0,ymax=1'))
    out['umbrales']=fig(''.join(a),'Diagnóstico de umbral en validación: medias de F1 y brecha TPR para LR. La selección del umbral no utiliza la prueba.','umbrales')
    z=S[S.analysis=='corrupcion'].copy();z['level']=z.setting.astype(float);z=z.groupby('level').mean(numeric_only=True)
    out['intensidad']=fig(axis([(z.index,z.f1,'F1'),(z.index,z.gap,'Brecha TPR')],xlabel='Multiplicador de corrupción',width=r'.8\textwidth',extra='ymin=0,ymax=1'),'Sensibilidad a la intensidad de corrupción: LR original, C fijo en 1 y retención constante. Cada punto requiere reentrenar; no es una modificación posterior de predicciones.','intensidad')
    series=[]
    for k in ['limpieza_rw','auditoria_rw']:series.append((range(3),[S[(S.analysis=='rubrica')&(S.setting==r+'_'+k)].f1.mean() for r in ['laxa','estricta','igual']],'M'+str(METHODS.index(k))))
    out['rubricas']=fig(axis(series,['Umbral 5','Umbral 7','Pesos iguales'],ylabel='F1',width=r'.8\textwidth',extra='ymin=0,ymax=1'),'Robustez frente a tres rúbricas alternativas. LR con C=1, reentrenado en cada condición. La prevalencia de positivos cambia respecto al experimento principal.','rubricas')
    c=pd.read_csv(R/'datos/cvs_11.csv.gz');ages=c.age.value_counts().sort_index();length=c.iloc[::6].text.str.split().str.len();hh,edges=np.histogram(length,bins=12)
    out['diversidad']=fig(axis([(ages.index,ages.values,'')],xlabel='Edad',ylabel='CVs',extra='ybar,ymin=0')+axis([((edges[:-1]+edges[1:])/2,hh,'')],xlabel='Palabras',ylabel='Perfiles',extra='ybar,ymin=0'),'Diversidad observable de la réplica 11: edades y longitud textual de un CV por perfil. El número de palabras no mide por sí solo realismo lingüístico.','diversidad')
    e=pd.read_csv(R/'explicaciones.csv');a=[]
    for k in ['original','auditoria_rw']:
        z=e[(e.method==k)&(e.kind=='global')].sort_values('value',key=abs).tail(7).sort_values('value')
        opts=r'width=.46\textwidth,height=4.5cm,xbar,xlabel={Coeficiente},ytick={'+','.join(str(i) for i in range(7))+r'},yticklabels={'+','.join(esc(s) for s in z.term)+r'},tick label style={font=\tiny},title={M'+str(METHODS.index(k))+r'},title style={font=\small}'
        a.append(r'\begin{tikzpicture}\begin{axis}['+opts+r']\addplot[fill=blue!50] coordinates {'+coords(z.value,range(7))+r'};\end{axis}\end{tikzpicture}')
    out['coeficientes']=fig(''.join(a),'Siete coeficientes de mayor magnitud entre los registrados, LR, semilla 11. El signo indica asociación aprendida; la magnitud depende de TF--IDF y la regularización y no representa efecto causal.','coeficientes')
    ci=pd.read_csv(R/'intervalos_pareados.csv');a=[]
    for metric in ['f1','gap']:
        z=ci[(ci.model=='LR')&(ci.metric==metric)];opts=r'width=.46\textwidth,height=3.8cm,ylabel={'+('Cambio F1' if metric=='f1' else 'Cambio brecha')+r'},xtick={0,1,2},xticklabels={M2,M4,M5},tick label style={font=\scriptsize},error bars/y dir=both,error bars/y explicit'
        txt=r'\begin{tikzpicture}\begin{axis}['+opts+r']\addplot+[only marks,error bars/.cd,y dir=both,y explicit] coordinates {'
        for j,(_,r) in enumerate(z.iterrows()):txt+=f'({j},{r.delta:.6f}) += (0,{r.high-r.delta:.6f}) -= (0,{r.delta-r.low:.6f}) '
        a.append(txt+r'};\addplot[gray,dashed,domain=-.3:2.3] {0};\end{axis}\end{tikzpicture}')
    out['intervalos']=fig(''.join(a),r'Diferencias pareadas respecto al original en LR. Intervalos bootstrap percentiles del 95\%, remuestreando cinco corpus completos 10.000 veces. Alcance exploratorio; no representan una población laboral.','intervalos')
    return out

def main():
    f=figures();base=(ROOT/'marco.tex').read_text(encoding='utf-8');method=(ROOT/'metodologia.tex').read_text(encoding='utf-8')
    method=method.replace('@@DIAGRAMA_PROCESO@@',diagram(['Generación independiente de perfiles y rúbrica $y^*$','Partición por perfil: entrenamiento 60\\%, validación 20\\%, prueba 20\\%','Solo entrenamiento: retención y corrupción $y^*\\rightarrow\\tilde y$','Tratamiento + TF--IDF + LR/SVM; C seleccionado en validación','Prueba intacta: rendimiento, grupos, pares y formato reservado'],'Flujo experimental y separación de información. La referencia limpia de prueba no interviene en el ajuste.','proceso'))
    method=method.replace('@@DIAGRAMA_CV@@',diagram(['Perfil profesional: competencias, experiencia e inglés $\\rightarrow$ rúbrica','Diversidad no puntuable: titulación, sector, proyecto, entregable y estilo','Seis variantes: género $\\times$ edad; cualificación constante','Señales explícitas en cabecera + señales indirectas en el cuerpo'],'Construcción de CVs emparejados y variación controlada.','cv'))
    method=method.replace('@@DIAGRAMA_MITIGACION@@',diagram(['Cabecera: retirar nombre, género y edad','Cuerpo: neutralizar flexión y eliminar año de finalización','Etiquetas: reparar el 30\\% de perfiles auditados de entrenamiento','Recalcular pesos grupo--etiqueta y ajustar el modelo'],'Componentes del tratamiento con auditoría. La reparación usa la rúbrica conocida y constituye un supuesto favorable.','mitigacion'))
    rows=[]
    for model in ['LR','SVM']:
        for k,n in zip(METHODS,NAMES):
            z=M[(M.model==model)&(M.method==k)&(M.scenario=='combinado')&(M.domain=='iid')]
            rows.append(model+' & '+n+' & '+num(z.f1.mean())+r' $\pm$ '+num(z.f1.std())+' & '+num(z.gap.mean())+r' $\pm$ '+num(z.gap.std())+' & '+num(z.auc.mean())+r'\\')
    table=r'\begin{table}[htbp]\centering\scriptsize\caption{Escenario combinado: media y desviación estándar de cinco corpus. AUC se expresa como media.}\label{tab:principal}\begin{tabular}{llrrr}\toprule Modelo & Tratamiento & F1 & Brecha TPR & AUC\\\midrule '+'\n'.join(rows)+r'\bottomrule\end{tabular}\end{table}'
    result=r'''\chapter{Resultados experimentales y explicación}
\label{cap:resultados}
\section{Ejecución efectiva y composición del corpus}
Se completaron cinco corpus independientes: 10.000 perfiles profesionales y 60.000 CVs principales. Sus variantes de formato se conservan por separado dentro de cada registro. Se realizaron 800 ajustes, de los cuales 750 corresponden a selección entre tres valores de C y 50 a sensibilidad. Se seleccionaron 250 modelos principales y se evaluaron en dos formatos, obteniendo 500 filas de métricas y 1.200.000 predicciones individuales. El número de documentos no se interpreta como tamaño de muestra independiente.

La comprobación separada recalculó F1, AUC y brechas a partir de las predicciones; verificó separación de perfiles, consistencia de la rúbrica, selección de C y corrección restringida a perfiles auditados. La ejecución y las versiones están documentadas en el recurso reproducible \cite{repositorio2026}. Esta verificación comprueba consistencia interna; no valida que la rúbrica represente productividad ni que las plantillas sean representativas de CVs reales.
@@diversidad@@
La figura~\ref{fig:diversidad} muestra variabilidad efectiva, no solo posibilidades teóricas del generador. Las edades cubren intervalos y las longitudes varían por perfil. Persisten estructuras repetidas y un vocabulario deliberadamente reducido. La figura~\ref{fig:datos} permite comprobar que el mecanismo de retención produce desigualdad de tamaño y que la corrupción actúa sobre positivos de referencia. La segunda proporción usa como denominador positivos retenidos, no todos los CVs.
@@datos@@
\section{Efectos del escenario antes de mitigar}
@@escenarios@@
En el control, LR obtuvo F1 de @@CONTROL_LR@@ y SVM de @@CONTROL_SVM@@. En el combinado, las cifras fueron @@ORIGINAL_LR@@ y @@ORIGINAL_SVM@@. La comparación debe leerse junto a las brechas: acertar globalmente no informa de cómo se distribuyen los rechazos erróneos. El control de tamaño mantiene la cantidad del combinado con etiquetas correctas y composición aproximadamente equilibrada. Por ello, una diferencia respecto a ese control no puede atribuirse simplemente a disponer de menos documentos.

Representación y etiquetado son mecanismos distintos. Retener menos candidaturas de un grupo reduce su presencia, pero no convierte por sí mismo competencias válidas en decisiones negativas. La corrupción sí introduce una asociación entre señales demográficas y rechazo de perfiles que cumplen la rúbrica. En el combinado ambos mecanismos interactúan; no es correcto sumar aritméticamente las brechas de los escenarios aislados para predecir su efecto conjunto.

\section{Comparación de mitigaciones y distribución de errores}
@@TABLE@@
@@comparacion@@
El tratamiento completo con auditoría produjo F1 de @@AUDIT_LR@@ en LR y @@AUDIT_SVM@@ en SVM, frente a @@ORIGINAL_LR@@ y @@ORIGINAL_SVM@@ sin tratamiento. Las diferencias respecto al original fueron @@DELTA_LR@@ y @@DELTA_SVM@@. Estas comparaciones describen este protocolo; no demuestran que corregir el 30\% de etiquetas siempre produzca la misma mejora.

Retirar solo la cabecera dejó una brecha TPR de @@HEADER_GAP@@ en LR. El resultado ilustra por qué una transformación superficial puede ser insuficiente: el modelo aún recibe una flexión de género y fechas asociadas con edad. La limpieza completa retira las señales que el propio generador conoce. Cuando las seis variantes quedan idénticas, la brecha es necesariamente cero sobre un conjunto que replica la misma cualificación en cada grupo. Este cero debe explicarse como invariancia por construcción y no como descubrimiento de una equidad universal.
@@grupos@@
Cada celda de la figura~\ref{fig:grupos} representa la proporción de positivos de referencia recuperados. Por ejemplo, TPR 0,70 significa que se descarta erróneamente el 30\% de quienes cumplen la rúbrica de ese grupo. Leer las seis celdas evita que promedios agregados por género oculten combinaciones de género y edad. Los tamaños de prueba y la prevalencia son iguales por construcción; esa simetría favorece interpretar las diferencias, pero no refleja necesariamente un proceso de selección real.

La reponderación interviene sobre la asociación de grupo y etiqueta observada. Si la etiqueta ya contiene errores, modificar pesos puede redistribuir su influencia sin conocer su verdad. La comparación entre limpieza con pesos y limpieza con pesos y auditoría aísla el efecto incremental de sustituir una parte de esas etiquetas en el mismo texto y sobre los mismos perfiles. No se atribuye toda la ganancia del tratamiento completo a un único componente. En SVM, añadir pesos a la limpieza reduce ligeramente la F1 media de 0,862 a 0,858, mientras que incorporar la auditoría la eleva a 0,921. En LR, la secuencia es 0,845, 0,855 y 0,910. Por tanto, combinar técnicas no garantiza una mejora monótona; en este diseño la reparación de etiquetas aporta la ganancia incremental más clara.
@@compromiso@@
La figura~\ref{fig:compromiso} debe leerse con dos criterios simultáneos. Una menor brecha es deseable, pero puede acompañarse de rechazo generalizado; una F1 alta puede esconder errores desigualmente repartidos. El punto de auditoría combina una referencia de evaluación limpia con reparación perfecta de una muestra de entrenamiento. Es una estimación bajo acceso favorable a información, no una promesa de mejora gratuita en producción.

\section{Invariancia emparejada, ordenación y umbral}
En LR original, el cambio de decisión entre variantes de género fue @@GENDER@@ y entre variantes de edad @@AGE@@. Se trata de proporciones de pares, manteniendo cualificación y redacción no demográfica. Las versiones limpias producen decisiones idénticas dentro de cada perfil; pueden seguir cometiendo errores sobre la cualificación común. Esta distinción entre invariancia e identificación correcta es esencial para interpretar la mitigación.
@@curvas@@
Las curvas de la réplica 11 ilustran la capacidad de ordenar candidaturas a distintos umbrales. No sustituyen la tabla agregada ni garantizan paridad: una curva global puede ser buena y coexistir con un umbral perjudicial para un subgrupo. La precisión--recall utiliza la prevalencia artificial del 50\%; su precisión no se extrapola a un proceso donde solo una fracción pequeña cumple los requisitos.
@@umbrales@@
Cambiar el umbral modifica el compromiso entre falsos negativos y falsos positivos. Los perfiles con puntuación próxima al corte son especialmente sensibles. La prueba auxiliar seleccionó el umbral solo en validación: F1 de prueba media @@THRESHOLD_ORIG@@ para LR original y @@THRESHOLD_AUDIT@@ para la versión con auditoría. Estas cifras no reemplazan el análisis principal con corte cero. Escoger un umbral tras observar la prueba habría convertido la evaluación en otra etapa de ajuste.

\section{Cambio de formato y sensibilidad del diseño}
@@formatos@@
Con formato reservado, la F1 de LR original fue @@OOD_ORIG@@ y la del tratamiento con auditoría @@OOD_AUDIT@@. La diferencia entre los dos formatos cuantifica sensibilidad a las modificaciones implementadas. Una estabilidad alta solo confirma tolerancia a ese cambio moderado: permanecen términos de competencias, reglas y estructura semántica próximos a los del entrenamiento.
@@intensidad@@
La prueba de intensidad mantiene constantes los CVs retenidos y aumenta la probabilidad de convertir positivos en negativos. Es una intervención sobre la supervisión, no sobre la capacidad profesional. La curva permite comprobar si las conclusiones dependen de una sola configuración de corrupción. Las pendientes no constituyen una ley extrapolable a tasas de discriminación reales; la familia de modelos, la regularización y la selección sintética de casos condicionan su forma.
@@rubricas@@
Las rúbricas alternativas modifican qué significa ser un candidato positivo. La comparación reentrena los modelos y evita evaluar predicciones antiguas contra un objetivo nuevo como si fueran modelos optimizados para él. Una F1 distinta puede reflejar tanto dificultad como cambio de prevalencia. El análisis no resuelve la validez normativa de la rúbrica: muestra cuánto dependen las conclusiones de una decisión profesional simplificada.

\section{Explicación de las predicciones e incertidumbre}
En un clasificador lineal, el margen se descompone exactamente como $s(x)=b+\sum_j w_jx_j$. El coeficiente global $w_j$ informa del signo aprendido; la contribución local es $w_jx_j$ y depende de que el término aparezca y de su peso TF--IDF. Los términos correlacionados pueden repartirse influencia. Ninguna de estas cantidades es una explicación causal de la idoneidad de una persona.
Como ejemplo verificable, las variantes 348 y 353 del perfil 58 tienen etiqueta positiva. LR original les asigna márgenes 3,043 y 1,956: ambas pasan el corte, aunque reciben puntuaciones distintas. Tras limpieza, pesos y auditoría, ambas obtienen 1,034. En el primer caso original, el margen se reconstruye como $-1{,}067+4{,}110=3{,}043$; en la versión auditada, como $0{,}391+0{,}643=1{,}034$. Las sumas se muestran redondeadas; la comprobación utiliza precisión completa. Este ejemplo no se seleccionó por cambiar la decisión y permite observar que igualdad de decisiones e igualdad de puntuaciones son propiedades distintas.
@@coeficientes@@
La inspección compara vocabulario profesional y posibles señales ajenas a la rúbrica. Eliminar un término del gráfico no prueba que toda información demográfica haya desaparecido: esa garantía solo existe aquí porque se conocen y verifican todas las señales del generador. El material reproducible conserva los modelos representativos y aportaciones locales, de forma que una predicción puede reconstruirse incluyendo intercepto y todos sus términos, sin depender de una explicación textual generada a posteriori.
@@intervalos@@
Los intervalos de la figura~\ref{fig:intervalos} usan diferencias pareadas entre corpus. Para LR, la mejora de F1 del tratamiento completo frente al original es 0,127, con intervalo exploratorio [0,116; 0,139]. En cambio, el cambio de brecha al retirar solo la cabecera es $-0,024$, con intervalo [$-0,056$; 0,003], que incluye cero. Este contraste desaconseja presentar la supresión de cabecera como una reducción robusta de disparidad en estas cinco réplicas. Con solo cinco réplicas, son sensibles a sus valores y tienen carácter exploratorio. Un intervalo degenerado en cero para una métrica estructural no demuestra ausencia de incertidumbre sobre CVs reales. La principal incertidumbre externa procede precisamente de lo que el generador no representa.
'''
    replacements={'CONTROL_LR':mean('original',scenario='control'),'CONTROL_SVM':mean('original',model='SVM',scenario='control'),'ORIGINAL_LR':mean('original'),'ORIGINAL_SVM':mean('original',model='SVM'),'AUDIT_LR':mean('auditoria_rw'),'AUDIT_SVM':mean('auditoria_rw',model='SVM'),'DELTA_LR':mean('auditoria_rw')-mean('original'),'DELTA_SVM':mean('auditoria_rw',model='SVM')-mean('original',model='SVM'),'HEADER_GAP':mean('cabecera','gap'),'GENDER':mean('original','gender_flip'),'AGE':mean('original','age_flip'),'THRESHOLD_ORIG':S[(S.analysis=='umbral')&(S.setting=='original')].f1.mean(),'THRESHOLD_AUDIT':S[(S.analysis=='umbral')&(S.setting=='auditoria_rw')].f1.mean(),'OOD_ORIG':mean('original',domain='ood'),'OOD_AUDIT':mean('auditoria_rw',domain='ood')}
    for k,v in replacements.items():result=result.replace('@@'+k+'@@',num(v))
    for k,v in f.items():result=result.replace('@@'+k+'@@',v)
    result=result.replace('@@TABLE@@',table)
    discussion=(ROOT/'discusion_anexos.tex').read_text(encoding='utf-8')
    examples=json.loads((R/'ejemplos_cvs.json').read_text(encoding='utf-8'));ex=[]
    for d in examples:
        ex.append(r'\subsection*{CV '+str(d['cv_id'])+'; perfil '+str(d['profile_id'])+'; puntuación '+str(d['reference_score'])+'; etiqueta '+str(d['y_true'])+'}\n'+r'\begin{quote}\small '+esc(d['text']).replace('\n',r'\par ')+'\n'+r'\end{quote}')
    discussion=discussion.replace('@@EJEMPLOS@@','\n'.join(ex))
    bib=(ROOT/'bibliografia.tex').read_text(encoding='utf-8')
    final=base+method+result+discussion.replace('@@BIBLIOGRAFIA@@',bib)+r'\end{document}'
    assert '@@' not in final
    (ROOT/'TFM_Jordan_Rajo_Vazquez_mejorado.tex').write_text(final,encoding='utf-8')
    print('Memoria generada:',len(final),'caracteres')

if __name__=='__main__':main()
