"""Generador transparente. Solo personas, empresas y trayectorias ficticias.

Las variantes comparten cualificación y estilo. El año de finalización y la
flexión de adjetivos son señales demográficas deliberadas, sin valor en la rúbrica.
"""
import re
import numpy as np

GROUPS=['H 24-34','H 38-48','H 52-62','M 24-34','M 38-48','M 52-62']
NAMES=[['Daniel','Pablo','Javier','Sergio','Mario','Adrián','Diego','Álvaro','Marcos','David','Hugo','Rubén','Iván','Lucas','Jorge','Raúl'],['Laura','Marta','Lucía','Elena','Sara','Clara','Paula','Irene','Julia','Carmen','Alba','Inés','Nuria','Eva','Raquel','Ana']]
SURNAMES=['García','López','Martín','Pérez','Sánchez','Romero','Ruiz','Torres','Navarro','Molina','Ortega','Ramos']
DEGREES=['grado en matemáticas','grado en economía','grado en ingeniería informática','formación profesional en administración','grado en estadística','grado en física']
SECTORS=['logística','comercio','energía','industria','servicios','turismo','educación','telecomunicaciones']
EXTRAS=['Excel','Git','Linux','documentación','comunicación','trabajo en equipo','control de versiones','presentaciones','calidad de datos','ofimática','gestión de incidencias','hojas de cálculo']
SKILLS={
 'sql':['SQL y consultas relacionales','consultas de bases de datos con SQL','SQL para combinar tablas'],
 'python':['Python para automatizar análisis','análisis de datos con Python','programación en Python'],
 'viz':['Power BI y cuadros de mando','visualización de indicadores con Power BI','informes visuales en Power BI'],
 'stats':['estadística aplicada e inferencia','análisis estadístico e intervalos de confianza','estadística descriptiva y contrastes']}

def rubric(p,variant='base'):
    if variant=='igual': return int(sum(p[k] for k in SKILLS)+int(p['years']>=2)+int(p['english']!='B1')>=4)
    q=2*p['sql']+2*p['python']+p['viz']+p['stats']+int(p['years']>=2)+int(p['english']!='B1')
    return int(q >= {'base':6,'laxa':5,'estricta':7}[variant])

def body(p,gender,age,style):
    """12 órdenes/formulaciones. 0-7 en desarrollo; 8-11 solo prueba OOD."""
    skill=', '.join(p['phrases']) if p['phrases'] else 'soporte documental y revisión de registros'
    extra=', '.join(p['extras']); grad=2026-age+20
    adj='organizado' if gender==0 else 'organizada'
    snippets=[
      f"PERFIL. Profesional {adj}, con interés en resolver necesidades de información del sector de {p['sector']}. Busco una posición de análisis con objetivos definidos y revisión de resultados.",
      f"EXPERIENCIA. Experiencia relevante: {p['years']} años. He trabajado en {p['jobs']} equipos de empresas ficticias, preparando entregables, revisando inconsistencias y documentando procesos para otras áreas.",
      f"COMPETENCIAS. {skill}. Herramientas complementarias: {extra}. Distingo los requisitos del encargo y comunico las limitaciones de los datos disponibles.",
      f"FORMACIÓN. {p['degree']}. Finalización: {grad}. Formación complementaria de {p['course_hours']} horas en organización de información; no constituye certificación adicional de las competencias anteriores.",
      f"IDIOMAS. Inglés: {p['english']}. Utilizo documentación de trabajo y preparo resúmenes para el equipo, dentro del nivel indicado.",
      f"PROYECTO. {p['project']}. La entrega consistió en {p['deliverable']}. El caso se describe para ilustrar responsabilidades; no se atribuyen resultados económicos ni títulos adicionales.",
      f"FORMA DE TRABAJO. Prefiero instrucciones verificables y revisiones periódicas. Modalidad: {p['mode']}. Disponibilidad: {p['availability']}."
    ]
    orders=[[0,1,2,3,4,5,6],[3,2,1,5,4,0,6],[0,5,2,1,4,3,6],[2,1,3,4,5,6,0],[1,5,2,3,0,4,6],[4,3,2,1,5,0,6],[5,0,1,2,3,4,6],[0,3,4,1,2,6,5],
            [5,2,4,0,3,6,1],[4,6,1,3,2,5,0],[1,4,5,0,6,2,3],[2,5,3,0,1,6,4]]
    if style>=8:
        snippets=[s.replace('PERFIL.','Acerca de mí:').replace('EXPERIENCIA.','Trayectoria:').replace('COMPETENCIAS.','Conocimientos:').replace('FORMACIÓN.','Estudios:').replace('IDIOMAS.','Lenguas:').replace('PROYECTO.','Caso de trabajo:').replace('FORMA DE TRABAJO.','Preferencias:').replace('Experiencia relevante:', 'Experiencia en análisis:').replace('Finalización:', 'Año de titulación:') for s in snippets]
    # Longitud y presencia de secciones no puntuables varían por perfil, no por grupo.
    ids=[i for i in orders[style] if i!=6 or p['long']]
    separator=['\n\n','\n','\n- ','\n\n'][style%4]
    return separator.join(snippets[i] for i in ids)

def suppress_header(text):
    return text.split('\n',1)[1]

def clean_text(text):
    """Reglas léxicas aplicadas al texto, sin leer perfil, grupo ni etiqueta."""
    text=suppress_header(text)
    text=re.sub(r'\borganizad[oa]\b','con organización',text,flags=re.I)
    text=re.sub(r'(?:Finalización|Año de titulación):\s*\d{4}\.','',text,flags=re.I)
    return text

def generate(n,seed):
    rng=np.random.default_rng(seed); profiles=[]; rows=[]; counts=[0,0]; seen=set()
    while len(profiles)<n:
        p={k:int(rng.random()<.55) for k in SKILLS}
        p.update(years=int(rng.integers(1,9)),english=str(rng.choice(['B1','B2','C1'])),degree=str(rng.choice(DEGREES)),sector=str(rng.choice(SECTORS)),
                 jobs=int(rng.integers(1,4)),course_hours=int(rng.choice([20,40,60,80,120])),mode=str(rng.choice(['presencial','híbrida','remota'])),availability=str(rng.choice(['inmediata','un mes','dos meses'])),
                 extras=rng.choice(EXTRAS,size=int(rng.integers(2,6)),replace=False).tolist(),long=bool(rng.integers(2)),
                 project=str(rng.choice(['Seguimiento de pedidos','Revisión de inventario','Consolidación de registros','Control de incidencias','Estudio de indicadores operativos','Organización de informes'])),
                 deliverable=str(rng.choice(['un resumen de incidencias','una propuesta de organización documental','un informe de seguimiento','una guía de comprobaciones','un registro de entregas'])))
        key=str(p); y=rubric(p)
        if counts[y]>=n//2 or key in seen: continue
        seen.add(key); counts[y]+=1
        p.update(profile_id=len(profiles),y_true=y,score=2*p['sql']+2*p['python']+p['viz']+p['stats']+int(p['years']>=2)+int(p['english']!='B1'),style=int(rng.integers(8)))
        p['phrases']=[str(rng.choice(SKILLS[k])) for k in SKILLS if p[k]]
        rng.shuffle(p['phrases']); profiles.append(p)
        ages=[int(rng.integers(a,b+1)) for a,b in [(24,34),(38,48),(52,62)]]
        names=[str(rng.choice(x))+' '+str(rng.choice(SURNAMES)) for x in NAMES]
        for g in range(6):
            sex=g//3; age=ages[g%3]
            header=f"Nombre ficticio: {names[sex]}. Género: {'hombre' if sex==0 else 'mujer'}. Edad: {age} años."
            text=header+'\n'+body(p,sex,age,p['style'])
            ood=header+'\n'+body(p,sex,age,8+p['profile_id']%4)
            rows.append(dict(cv_id=len(rows),profile_id=p['profile_id'],group=g,age=age,gender=sex,y_true=y,text=text,text_ood=ood))
    for start in range(0,len(rows),6):
        assert len({clean_text(r['text']) for r in rows[start:start+6]})==1
        assert len({suppress_header(r['text']) for r in rows[start:start+6]})==6
    assert len({clean_text(rows[i*6]['text']) for i in range(n)})==n
    return profiles,rows
