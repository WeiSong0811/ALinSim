# -*- coding: utf-8 -*-
import sys;
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#Add parent directory
if('..' not in sys.path):
    sys.path.insert(0,'..')

import Finite_Element_Plate as fep;
import Graphics as fep_graphics;
from Generate_Floor_Mesh import read_text_file;

files_path = f'./dataset_txt'
import os

files = [os.path.join(files_path, file) for file in os.listdir(files_path) if file.endswith('.txt')]
from tqdm import tqdm

for file in tqdm(files, desc="Running slab analysis"):
    tqdm.write(f"Processing: {os.path.basename(file)}")
    #Test read_text_file
    file_name=file;
    file_name_use = os.path.splitext(os.path.basename(file))[0]
    output_path = f'./output/{file_name_use}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    (nodes,elements,sections,materials,global_freedoms,load_cases,
    load_combos,criteria,mesh_input,area_loads,node_loads)=read_text_file(file_name);

    #Test run analysis
    analysis_output=fep.run_analysis(nodes,elements,sections,materials,
                                    node_loads,load_cases);
    #Seperate output of analysis output
    nodes=analysis_output[0];
    elements=analysis_output[1];
    k_global=analysis_output[2];
    k_global_red=analysis_output[3];
    p_global=analysis_output[4];
    p_global_red=analysis_output[5];
    p_global_post=analysis_output[6];
    u_global=analysis_output[7];
    u_global_red=analysis_output[8];

    #Test print_load_and_reaction_summary
    results = fep.print_load_and_reaction_summary(load_cases,nodes,p_global,p_global_post);

    #Test plot_mesh_2d
    fig,ax=fep_graphics.plot_mesh_2d(nodes,elements);
    fig.savefig(f'./{output_path}/Slab_Example_mesh.png',dpi=300);
    plt.close(fig);

    import numpy as np
    import pandas as pd
    for case in load_cases:
        np.savetxt(f"./{output_path}/p_global_{case}.csv", p_global[case], delimiter=",")
        np.savetxt(f"./{output_path}/u_global_{case}.csv", u_global[case], delimiter=",")
        np.savetxt(f"./{output_path}/p_global_post_{case}.csv", p_global_post[case], delimiter=",")

    df = pd.DataFrame(results)
    df['slab_id'] = file_name_use
    df.to_csv(f"./{output_path}/Slab_Example_summary.csv", index=False)