from pathlib import Path
import pandas as pd

def slab(b,h,d,num_material,orientation_angle,automesh):
    point_lu = (0,0)
    point_ru = (b,0)
    point_rt = (b,h)
    point_lt = (0,h)

    slab_points = (point_lu, point_ru, point_rt, point_lt)

    return f'MESH,SLAB,{slab_points},{d},{num_material},{orientation_angle},{automesh}'

def support(b,h,support_x,support_y):
    
    if support_x >= b or support_y >= h:
        raise ValueError("Support position exceeds slab dimensions.")
    
    return f'MESH,PT SUPPORT,{support_x},{support_y}'

def material(num_material,elastic_modulus,poisson_ratio,density):
    return f'MATERIAL,{num_material},{elastic_modulus},{poisson_ratio},{density}'

def orientataion(latitude,longitude):
    return f'ORIENTATION,LATITUDE,{latitude}\nORIENTATION,LONGITUDE,{longitude}'

def load():
    dl = 'LOAD CASE,DL,DEAD\n'
    sdl = 'LOAD CASE,SDL,OTHER DEAD\n'
    ll = 'LOAD CASE,LL,LIVE\n'
    llred = 'LOAD CASE,LLRED,LIVE REDUCIBLE'
    return dl + sdl + ll + llred

def mesh_size(size):
    return f'MESH SIZE,{size}'

def loading(sdl, ll):
    dl = f'LOADING,DL,GRAVITY\n'
    sdl = f'LOADING,SDL,{sdl},ALL\n'
    ll = f'LOADING,LL,{ll},ALL'
    return dl + sdl + ll

def combo():
    return (
        'LOAD COMBO,MAX SERVICE,((DL,1.0),(SDL,1.0),(LL,1.0),(LLRED,1.0))\n'
        'LOAD COMBO,SERVICE,((DL,1.0),(SDL,1.0),(LL,0.7),(LLRED,0.7))\n'
        'LOAD COMBO,LONGTERM,((DL,1.0),(SDL,1.0),(LL,0.4),(LLRED,0.4))'
    )

def ultimate_strength():
    return (
        'LOAD COMBO,STRENGTH,((DL,0.9),(SDL,0.9))\n'
        'LOAD COMBO,STRENGTH,((DL,0.9),(SDL,0.9),(LL,1.5),(LLRED,1.5))\n'
        'LOAD COMBO,STRENGTH,((DL,1.35),(SDL,1.35))\n'
        'LOAD COMBO,STRENGTH,((DL,1.2),(SDL,1.2),(LL,1.5),(LLRED,1.5))'
    )

out_dir = Path(f"./Slab_Example_01/dataset_txt")
out_dir.mkdir(exist_ok=True)
results = []

df = pd.read_csv("FEA_inpute_pos.csv")

for i in range(df.shape[0]):
    b = df.iloc[i, df.columns.get_loc("b")]
    b = b.item()  # Convert from numpy scalar to Python scalar
    h_b = df.iloc[i, df.columns.get_loc("h/d")]
    h_b = h_b.item()  # Convert from numpy scalar to Python scalar
    d = df.iloc[i, df.columns.get_loc("d")]
    d = d.item()  # Convert from numpy scalar to Python scalar
    h = b * h_b
    E = df.iloc[i, df.columns.get_loc("E")]
    E = E.item()  # Convert from numpy scalar to Python scalar
    nu = 0.2
    rho = 2400

    sdl = df.iloc[i, df.columns.get_loc("sdl")]
    sdl = sdl.item()  # Convert from numpy scalar to Python scalar
    ll = df.iloc[i, df.columns.get_loc("ll")]
    ll = ll.item()  # Convert from numpy scalar to Python scalar

    mesh = 100
    angle = 90
    pos = df.iloc[i, df.columns.get_loc("position")]
    pos = pos.item()  # Convert from numpy scalar to Python scalar
    txt = []
    txt.append(slab(b, h, d, 32, 0, 1))
    supports = [
    (int(pos*b), int(pos*h)),
    (int((1-pos)*b), int(pos*h)),
    (int(pos*b), int((1-pos)*h)),
    (int((1-pos)*b), int((1-pos)*h)),
    ]
    # 支座（示例 4 点）
    for sx, sy in supports:
        txt.append(support(b, h, int(sx), int(sy)))

    txt.append(material(32, E, nu, rho))
    txt.append(orientataion(0, angle))
    txt.append(load())
    txt.append(mesh_size(mesh))
    txt.append(loading(sdl, ll))
    txt.append(combo())
    txt.append(ultimate_strength())

    (out_dir / f"slab_{i:03d}.txt").write_text("\n".join(txt))
    results.append({
        'slab_id': f'slab_{i:03d}',
        'b': b,
        'h': h,
        'd': d,
        'E': E,
        'nu': nu,
        'rho': rho,
        'sdl': sdl,
        'll': ll,
        'mesh': mesh,
        'angle': angle,
        'sx1': supports[0][0], 'sy1': supports[0][1],
        'sx2': supports[1][0], 'sy2': supports[1][1],
        'sx3': supports[2][0], 'sy3': supports[2][1],
        'sx4': supports[3][0], 'sy4': supports[3][1],
    })
results_df = pd.DataFrame(results)
results_df.to_csv(f"./Slab_Example_01/parameters.csv", index=False)