import numpy as np 
import csv
from pathlib import Path

def _is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating))

def _parse_spec(spec):
    if isinstance(spec, dict):
        if 'values' in spec:
            vals = spec['values']
            if len(vals) < 2:
                raise ValueError(f"Values list must have at least 2 items: {vals}")
            return {'kind': 'discrete', 'values': np.asarray(vals, dtype=object)}
        
        if 'low' in spec and 'high' in spec:
            low, high = spec['low'], spec['high']
            if not (_is_number(low)) and _is_number(high) and high > low:
                raise ValueError(f"Low and high must be numbers and low < high: low={low}, high={high}")
            scale = spec.get('scale', 'linear')
            dtype = spec.get('dtype', 'float')
            return {'kind': 'continuous', 'low': float(low), 'high': float(high), 'scale': scale, 'dtype': dtype}
        raise ValueError('dict spec must have either "values" or "low" and "high" keys')
    
    if isinstance(spec, tuple) and len(spec) == 2 and _is_number(spec[0]) and _is_number(spec[1]):
        low, high = spec
        if not (high > low):
            raise ValueError(f'In tuple spec, high must be greater than low: low={low}, high={high}')
        return {'kind': 'continuous', 'low': float(low), 'high': float(high), 'scale': 'linear', 'dtype': 'float'}
    
    if isinstance(spec, (list, np.ndarray)):
        if len(spec) < 2:
            raise ValueError(f"List spec must have at least 2 items: {spec}")
        return {'kind': 'discrete', 'values': np.asarray(spec, dtype=object)}
    
    raise ValueError(f"Invalid spec format: {spec}")

def lhs_mixed(param_specs, n_samples=200, seed=42):

    rng = np.random.default_rng(seed)
    specs = [_parse_spec(s) for s in param_specs]
    d = len(specs)
    n = int(n_samples)
    if d <= 0 or n <= 0:
        raise ValueError(f"Number of parameters and samples must be positive: d={d}, n={n}")
    
    samples = np.empty((n, d), dtype=object)

    for j, sp in enumerate(specs):
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n

        if sp['kind'] == 'continuous':
            low, high = sp['low'], sp['high']
            if sp.get('scale', 'linear') == 'log':
                if low <= 0 or high <= 0:
                    raise ValueError(f"Log scale requires positive low and high: low={low}, high={high}")
                log_low, log_high = np.log(low), np.log(high)
                x = np.exp(log_low + u * (log_high - log_low))

            else:
                x = low + u * (high - low)
            
            if sp.get('dtype', 'float') == 'int':
                x = np.rint(x).astype(int)

            samples[:, j] = x.astype(object)
        
        elif sp['kind'] == 'discrete':
            vals = sp['values']
            L = len(vals)

            order = np.argsort(u)
            base = n // L
            rem = n % L
            counts = np.full(L, base, dtype=int)
            counts[:rem] += 1

            levels = np.repeat(np.arange(L), counts)
            rng.shuffle(levels)
            idx = np.empty(n, dtype=int)
            idx[order] = levels

            samples[:, j] = vals[idx]
        
        else:
            raise ValueError(f"Unknown spec kind: {sp['kind']}")
        
    return samples

def save_csv(samples, headers, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if headers is None:
        headers = [f'p{i+1}' for i in range(samples.shape[1])]

    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(samples)

if __name__ == "__main__":
    N = 200
    seed = 42

    param_specs = [
        #(3/7, 1), # continuous uniform
        #(2/3, 1),
        #(0, 1),
        #(0.35, 1),
        #(0.5, 1),
        #(4/7, 1),
        #(0.311, 1),
        #{'low': 1, 'high': 100, 'dtype': 'int'},  # continuous integer
        [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250],  # discrete list
        [1, 1.25, 1.5, 1.75, 2, 2.25],
        [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
        [24000, 26700, 30100, 32800, 34800, 37400, 39600, 42200],
        [-1.4, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0],
        [0, -0.25, -1, -1.25, -1.5],
        [0.1, 0.15, 0.2, 0.25, 0.3]
        #{'values': ['red', 'green', 'blue']}  # discrete categorical
    ]
    samples = lhs_mixed(param_specs, n_samples=N, seed=seed)
    '''
    headers = ['PS:PAN ratio', 
               'Feed rate(mL/h)', 
               'Distance(cm)', 
               'Mass fraction of solute', 
               'Mass fraction of SiO2 in solute ', 
               'Applied voltage(kV)',
               'Inner diameter(mm)']
    '''
    headers = ['d',
               'h/d',
               'b',
               'E',
               'll',
               'sdl',
               'position'
               ]
    save_csv(samples, headers, 'FEA_inpute_pos.csv')

    print('LHS sampling completed and saved to FEA_inpute_pos.csv')
