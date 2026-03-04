# import Finite_Element_Plate as fep;
# from Generate_Floor_Mesh import read_parameters;
from . import Finite_Element_Plate as fep
from .Generate_Floor_Mesh import read_parameters as r_fea

import matplotlib;
matplotlib.use("Agg");
from tqdm import tqdm;
import pandas as pd

def _as_float(v):
    # Compatible with scalar, numpy scalar, and single-element pandas Series.
    if hasattr(v, 'iloc'):
        return float(v.iloc[0]);
    return float(v);

def build_params_from_row(row):
    b=_as_float(row['b']);
    h_over_d=_as_float(row['h/d']);
    d=_as_float(row['d']);
    h=b*h_over_d;
    E=_as_float(row['E']);
    sdl=_as_float(row['sdl']);
    ll=_as_float(row['ll']);

    # If CSV has no support position column, use 15% edge offset.
    pos=_as_float(row.get('position',0.2));
    supports=[
        (int(pos*b),int(pos*h)),
        (int((1-pos)*b),int(pos*h)),
        (int(pos*b),int((1-pos)*h)),
        (int((1-pos)*b),int((1-pos)*h)),
    ];

    return {
        'slab':{
            'points':[(0,0),(b,0),(b,h),(0,h)],
            'thk':int(round(d)),
            'grade':32,
            'toc':0,
            'priority':1,
        },
        'supports':supports,
        'material':{
            'name':32,
            'E':E,
            'poisson':0.2,
            'density':2400,
        },
        'orientation':{
            'latitude':0,
            'longitude':90,
        },
        'mesh_size':100,
        'load_cases':[
            {'name':'DL','type':'DEAD'},
            {'name':'SDL','type':'OTHER DEAD'},
            {'name':'LL','type':'LIVE'},
            {'name':'LLRED','type':'LIVE REDUCIBLE'},
        ],
        'loadings':[
            {'case':'DL','gravity':True},
            {'case':'SDL','loading':sdl,'areas':'ALL'},
            {'case':'LL','loading':ll,'areas':'ALL'},
        ],
        'load_combos':[
            {'name':'MAX SERVICE','factors':[('DL',1.0),('SDL',1.0),('LL',1.0),('LLRED',1.0)]},
            {'name':'SERVICE','factors':[('DL',1.0),('SDL',1.0),('LL',0.7),('LLRED',0.7)]},
            {'name':'LONGTERM','factors':[('DL',1.0),('SDL',1.0),('LL',0.4),('LLRED',0.4)]},
            {'name':'STRENGTH','factors':[('DL',0.9),('SDL',0.9)]},
            {'name':'STRENGTH','factors':[('DL',0.9),('SDL',0.9),('LL',1.5),('LLRED',1.5)]},
            {'name':'STRENGTH','factors':[('DL',1.35),('SDL',1.35)]},
            {'name':'STRENGTH','factors':[('DL',1.2),('SDL',1.2),('LL',1.5),('LLRED',1.5)]},
        ],
    };


def run_one(params):
    (nodes,elements,sections,materials,global_freedoms,load_cases,
     load_combos,criteria,mesh_input,area_loads,node_loads)=r_fea(params);

    analysis_output=fep.run_analysis(nodes,elements,sections,materials,node_loads,load_cases);

    u_global=analysis_output[7];

    return u_global;

def run_dataframe(df):
    required_cols=['d','h/d','b','E','ll','sdl'];
    missing=[c for c in required_cols if c not in df.columns];
    if(missing):
        raise ValueError(f'Missing required columns: {missing}. Existing columns: {list(df.columns)}');

    row=df.iloc[0];
    params=build_params_from_row(row);

    row_results=run_one(params);

    return row_results

def run_fea(x):
    import numpy as np
    # x = [170.0,1.0,500.0,24000.0,-1.5,0.0]
    cols=['d','h/d','b','E','ll','sdl']
    x = pd.DataFrame([x], columns=cols)
    u_all = run_dataframe(x)
    
    dl = u_all['DL'][0::3,0]
    sdl = u_all['SDL'][0::3,0]
    ll = u_all['LL'][0::3,0]
    llred = u_all['LLRED'][0::3,0]
    max_dl = np.max(np.abs(dl))
    max_sdl = np.max(np.abs(sdl))
    max_ll = np.max(np.abs(ll))
    max_llred = np.max(np.abs(llred))
    max_u = max(max_dl, max_ll, max_sdl, max_llred)

    return max_u