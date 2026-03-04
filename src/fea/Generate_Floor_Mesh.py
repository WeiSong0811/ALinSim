# -*- coding: utf-8 -*-
from .Engineering_Formulas import *;
from .Geometric_Function import *;
import os;
import tempfile;

def read_text_file(file_name):
    #Intiate the global freedoms
    global_freedoms={
        'x-translation':False,
        'y-translation':False,
        'z-translation':True,
        'x-rotation':True,
        'y-rotation':True,
        'z-rotation':False
        };
    #Initiate the nodes
    nodes={};
    #Initiate the elements
    elements={};
    #Inititate the sections
    sections={};
    #Initiate the materials
    materials={};
    #Initiate the load cases
    load_cases={};
    #Initiate the load combos
    load_combos={
        'max service':{'count':0,
                       'combinations':[]},
        'service':{'count':0,
                   'combinations':[]},
        'longterm':{'count':0,
                    'combinations':[]},
        'strength':{'count':0,
                    'combinations':[]}
        };
    #Initiate the criteria
    criteria={
        'mesh size':250.0,
        'latitude dir':0.0,
        'longitude dir':90.0,
        'design code':'AS3600-2018'
        };
    #Initiate the temporary imports
    mesh_input={
        'pt supports':0,
        'slabs':0
        };
    #Initiate the area loads
    area_loads={};
    #read the text file
    with open(file_name,mode='r') as file:
        input_text=[]; #List for storing the input text
        #Loop thru file and add lines to input text
        for line in file:
            input_text+=[line.strip('\n')];
    file.close();
    #Loop thru the input_text list and distribute to required dictionaries
    case_no=0;
    for i in range(len(input_text)):
        line_id=input_text[i].split(',',maxsplit=1);
        #condition statements
        if(line_id[0].upper()=='MESH'):
            mesh_type=line_id[1].split(',',maxsplit=1);
            if(mesh_type[0].upper()=='PT SUPPORT'):
                x_pt,y_pt=mesh_type[1].rsplit(',',2);
                x_pt=round(float(x_pt),4);
                y_pt=round(float(y_pt),4);
                mesh_input['pt supports']+=1;
                pt_int=mesh_input['pt supports'];
                mesh_input['PT_SUPPORT_'+str(pt_int)]={
                    'x_pt':x_pt,
                    'y_pt':y_pt};
            elif(mesh_type[0].upper()=='SLAB'):
                slab,thk,grade,toc,priority=mesh_type[1].rsplit(',',4);
                thk=int(thk);
                grade=int(grade);
                toc=int(toc);
                priority=int(priority);
                slab=slab.replace('(','');
                slab=slab.replace(')','');
                slab=slab.split(',');
                slab_pts=[];
                for j in range(len(slab)//2):
                    slab_pts+=[[round(float(slab[2*j]),3),
                                round(float(slab[2*j+1]),3)]];
                mesh_input['slabs']+=1;
                slab_int=mesh_input['slabs'];
                mesh_input['SLAB_'+str(slab_int)]={
                    'thk':thk,
                    'grade':grade,
                    'toc':toc,
                    'priority':priority,
                    'slab pts':slab_pts};
        elif(line_id[0].upper()=='MATERIAL'):
            mat_name,E,poisson,density=line_id[1].split(',');
            mat_name='CONC_'+mat_name;
            E=round(float(E),3);
            poisson=round(float(poisson),3);
            density=round(float(density),3);
            materials[mat_name]={
                'E':E,
                'Poisson':poisson,
                'density':density,
                };
        elif(line_id[0].upper()=='ORIENTATION'):
            des_ori,des_ang=line_id[1].split(',');
            des_ang=round(float(des_ang),3);
            if(des_ori.upper()=='LATITUDE'):
                criteria['latitude dir']=des_ang;
            elif(des_ori.upper()=='LONGITUDE'):
                criteria['longitude dir']=des_ang;
        elif(line_id[0].upper()=='MESH SIZE'):
            mesh_size=round(float(line_id[1]),3);
            criteria['mesh size']=mesh_size;
        elif(line_id[0].upper()=='LOAD CASE'):
            case_no+=1;
            case_name,case_type=line_id[1].split(',');
            load_cases[case_name]={
                'number':case_no,
                'case type':case_type
                };
        elif(line_id[0].upper()=='LOADING'):
            loading=line_id[1].split(',');
            case=loading[0];
            if(len(loading)==2 and loading[1].upper()=='GRAVITY'):
                area_loads[case]={'gravity':True};
            elif(len(loading)==3 and loading[2].upper()=='ALL'):
                area_loads[case]={
                    'areas':'all',
                    'loading':round(float(loading[1]),4)
                    };
        elif(line_id[0].upper()=='LOAD COMBO'):
            case_name,case_combos=line_id[1].split(',',maxsplit=1);
            load_combos[case_name.lower()]['count']+=1;
            case_combos=case_combos.replace('(','');
            case_combos=case_combos.replace(')','');
            case_combos=case_combos.split(',');
            case_combo=[];
            for j in range(len(case_combos)//2):
                case_combo+=[[case_combos[2*j],
                              round(float(case_combos[2*j+1]),3)]];
            load_combos[case_name.lower()]['combinations']+=[case_combo];
    #Generate nodes
    x_pts=[];
    y_pts=[];
    for item in mesh_input:
        if(item not in ['pt supports','slabs']):
            if('PT_SUPPORT_' in item):
                x_pt=mesh_input[item]['x_pt'];
                y_pt=mesh_input[item]['y_pt'];
                if(x_pt not in x_pts):
                    x_pts+=[x_pt];
                if(y_pt not in y_pts):
                    y_pts+=[y_pt];
            elif('SLAB_' in item):
                for i in range(len(mesh_input[item]['slab pts'])):
                    x_pt=mesh_input[item]['slab pts'][i][0];
                    y_pt=mesh_input[item]['slab pts'][i][1];
                    if(x_pt not in x_pts):
                        x_pts+=[x_pt];
                    if(y_pt not in y_pts):
                        y_pts+=[y_pt];
    x_pts.sort();
    y_pts.sort();
    mesh_size=criteria['mesh size'];
    x_pts_temp=[];
    for i in range(len(x_pts)-1):
        if((x_pts[i+1]-x_pts[i])%mesh_size==0):
            nos_divs=int((x_pts[i+1]-x_pts[i])/mesh_size);
        else:
            nos_divs=int((x_pts[i+1]-x_pts[i])/mesh_size)+1;
        mesh_split=round((x_pts[i+1]-x_pts[i])/nos_divs,1);
        for j in range(1,nos_divs):
            x_pts_temp+=[round(x_pts[i]+j*mesh_split,1)];
    x_pts+=x_pts_temp;
    x_pts.sort();
    y_pts_temp=[];
    for i in range(len(y_pts)-1):
        if((y_pts[i+1]-y_pts[i])%mesh_size==0):
            nos_divs=int((y_pts[i+1]-y_pts[i])/mesh_size);
        else:
            nos_divs=int((y_pts[i+1]-y_pts[i])/mesh_size)+1;
        mesh_split=round((y_pts[i+1]-y_pts[i])/nos_divs,1);
        for j in range(1,nos_divs):
            y_pts_temp+=[round(y_pts[i]+j*mesh_split,1)];
    y_pts+=y_pts_temp;
    y_pts.sort();
    #Generate the nodes dictionary
    node_count=0;
    for y_pt in y_pts:
        for x_pt in x_pts:
            node_count+=1;
            nodes[node_count]={
                'x':x_pt,'y':y_pt,
                'fix_z':False,'fix_mx':False,'fix_my':False
                };
    #Create the nodes index
    nodes_index=[];
    nodes_id=[];
    for node in nodes:
        nodes_index+=[[nodes[node]['x'],nodes[node]['y']]];
        nodes_id+=[node];
    #Create the elements
    slab_id=[];
    for item in mesh_input:
        if('SLAB_' in item):
            thk=mesh_input[item]['thk'];
            grade=mesh_input[item]['grade'];
            toc=mesh_input[item]['toc'];
            priority=mesh_input[item]['priority'];
            slab_pts=mesh_input[item]['slab pts'];
            mat='CONC_'+str(grade);
            section='S{}G{}'.format(thk,grade);
            sections[section]={
                'Thickness':thk,
                'Material':mat,
                };
            slab_id+=[item];
    #Create the elements dictionary
    element_no=0;
    for i in range(len(y_pts)-1):
        for j in range(len(x_pts)-1):
            xi,yi=x_pts[j],y_pts[i];
            xj,yj=x_pts[j+1],y_pts[i];
            xk,yk=x_pts[j+1],y_pts[i+1];
            xl,yl=x_pts[j],y_pts[i+1];
            xc,yc=0.5*(xj-xi),0.5*(yk-yi);
            in_mesh=False;
            for slab in slab_id:
                slab_polygon=mesh_input[slab]['slab pts'];
                if(ptInPolygon((xc,yc),slab_polygon)==True):
                    thk=mesh_input[slab]['thk'];
                    grade=mesh_input[slab]['grade'];
                    element_no+=1;
                    node_i=nodes_id[nodes_index.index([xi,yi])];
                    node_j=nodes_id[nodes_index.index([xj,yj])];
                    node_k=nodes_id[nodes_index.index([xk,yk])];
                    node_l=nodes_id[nodes_index.index([xl,yl])];
                    section='S{}G{}'.format(thk,grade);
                    elements[element_no]={
                        'Node_i':node_i,
                        'Node_j':node_j,
                        'Node_k':node_k,
                        'Node_l':node_l,
                        'Section':section
                        };
    #Create the loadings
    node_loads={};
    for node in nodes:
        node_loads[node]={};
        for case in load_cases:
            node_loads[node][case]=[0,0,0];
    #Get the element areas
    for ele in elements:
        node_i=elements[ele]['Node_i'];
        node_j=elements[ele]['Node_j'];
        node_k=elements[ele]['Node_k'];
        node_l=elements[ele]['Node_l'];
        area_pgs=[[nodes[node_i]['x'],nodes[node_i]['y']],
                  [nodes[node_j]['x'],nodes[node_j]['y']],
                  [nodes[node_k]['x'],nodes[node_k]['y']],
                  [nodes[node_l]['x'],nodes[node_l]['y']]];
        ele_area,xc,yc=polygonArea(area_pgs);
        elements[ele]['area']=ele_area;
        elements[ele]['xc']=xc;
        elements[ele]['yc']=yc;
    #Add the loading to the node loads
    for case in area_loads:
        if(area_loads[case].get('gravity',False)==True):
            for ele in elements:
                node_i=elements[ele]['Node_i'];
                node_j=elements[ele]['Node_j'];
                node_k=elements[ele]['Node_k'];
                node_l=elements[ele]['Node_l'];
                area=0.25*elements[ele]['area'];
                sect=elements[ele]['Section'];
                thickness=sections[sect]['Thickness'];
                mat=sections[sect]['Material'];
                density=materials[mat]['density'];
                fy=round(-9.81*density*thickness*area*10**-12,3);
                node_loads[node_i][case][0]=round(
                    node_loads[node_i][case][0]+fy,3);
                node_loads[node_j][case][0]=round(
                    node_loads[node_j][case][0]+fy,3);
                node_loads[node_k][case][0]=round(
                    node_loads[node_k][case][0]+fy,3);
                node_loads[node_l][case][0]=round(
                    node_loads[node_l][case][0]+fy,3);                
        if(area_loads[case].get('areas',False)=='all'):
            load=area_loads[case]['loading'];
            for ele in elements:
                node_i=elements[ele]['Node_i'];
                node_j=elements[ele]['Node_j'];
                node_k=elements[ele]['Node_k'];
                node_l=elements[ele]['Node_l'];
                area=0.25*elements[ele]['area'];
                fy=round(load*area*10**-6,3);
                node_loads[node_i][case][0]=round(
                    node_loads[node_i][case][0]+fy,3);   
                node_loads[node_j][case][0]=round(
                    node_loads[node_j][case][0]+fy,3);
                node_loads[node_k][case][0]=round(
                    node_loads[node_k][case][0]+fy,3);
                node_loads[node_l][case][0]=round(
                    node_loads[node_l][case][0]+fy,3);                
    #Add nodal restraint
    for item in mesh_input:
        if('PT_SUPPORT_' in item):
            node=nodes_index.index(
                [mesh_input[item]['x_pt'],mesh_input[item]['y_pt']])+1;
            nodes[node]['fix_z']=True;
                
    return (nodes,elements,sections,materials,global_freedoms,load_cases,
            load_combos,criteria,mesh_input,area_loads,node_loads);

def _combo_to_text(case_name, factors):
    combo_items=[];
    for case,factor in factors:
        combo_items+=['({},{})'.format(case,float(factor))];
    return 'LOAD COMBO,{},({})'.format(case_name,','.join(combo_items));

def parameters_to_input_text_lines(params):
    """
    Build solver input text lines from a parameter dictionary.

    Required keys:
        slab: {'points': [(x1,y1),...], 'thk': int, 'grade': int}
        material: {'name': int|str, 'E': float|int, 'poisson': float, 'density': float}
        load_cases: [{'name': str, 'type': str}, ...]
        loadings: [{'case': str, 'gravity': True} or
                   {'case': str, 'loading': float, 'areas': 'ALL'}, ...]
        load_combos: [{'name': str, 'factors': [(case, factor), ...]}, ...]

    Optional keys:
        slab.toc (default 0), slab.priority (default 1)
        supports: [(x,y), ...]
        orientation: {'latitude': float, 'longitude': float}
        mesh_size: float (default 250)
    """
    slab=params['slab'];
    material=params['material'];
    load_cases=params['load_cases'];
    loadings=params['loadings'];
    load_combos=params['load_combos'];
    supports=params.get('supports',[]);
    orientation=params.get('orientation',{'latitude':0,'longitude':90});
    mesh_size=params.get('mesh_size',250.0);

    toc=slab.get('toc',0);
    priority=slab.get('priority',1);
    slab_points=tuple((float(pt[0]),float(pt[1])) for pt in slab['points']);

    lines=[];
    lines+=['MESH,SLAB,{},{},{},{},{}'.format(
        slab_points,int(slab['thk']),int(slab['grade']),int(toc),int(priority)
    )];

    for sx,sy in supports:
        lines+=['MESH,PT SUPPORT,{},{}'.format(float(sx),float(sy))];

    lines+=['MATERIAL,{},{},{},{}'.format(
        material['name'],material['E'],material['poisson'],material['density']
    )];
    lines+=['ORIENTATION,LATITUDE,{}'.format(orientation.get('latitude',0))];
    lines+=['ORIENTATION,LONGITUDE,{}'.format(orientation.get('longitude',90))];
    lines+=['MESH SIZE,{}'.format(float(mesh_size))];

    for case in load_cases:
        lines+=['LOAD CASE,{},{}'.format(case['name'],case['type'])];

    for ld in loadings:
        if(ld.get('gravity',False)==True):
            lines+=['LOADING,{},GRAVITY'.format(ld['case'])];
        else:
            areas=str(ld.get('areas','ALL')).upper();
            lines+=['LOADING,{},{},{}'.format(ld['case'],float(ld['loading']),areas)];

    for combo in load_combos:
        lines+=[_combo_to_text(combo['name'],combo['factors'])];

    return lines;

def read_parameters(params):
    """
    Direct parameter interface. No user-managed txt file is required.
    """
    lines=parameters_to_input_text_lines(params);
    tmp_path=None;
    try:
        with tempfile.NamedTemporaryFile(mode='w',suffix='.txt',delete=False) as tmp:
            tmp.write('\n'.join(lines));
            tmp_path=tmp.name;
        return read_text_file(tmp_path);
    finally:
        if(tmp_path is not None and os.path.exists(tmp_path)):
            os.remove(tmp_path);

if(__name__=='__main__'):
    #Get the filename to be imported
    file_name=input('Enter the file path for the text file: ');

    (nodes,elements,sections,materials,global_freedoms,load_cases,
     load_combos,criteria,mesh_input,area_loads,node_loads)=read_text_file(file_name);
