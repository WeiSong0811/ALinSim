import pandas as pd
import numpy as np
from fea import run_fea

for seed in range(30,40):
    print('Analyse:',seed)
    csv_file = f'../data/fea_test_{seed}.csv'
    df = pd.read_csv(csv_file)

    num = df.shape[0]

    max_all_list = []
    for i in range(num):
        print('Run: [',i ,'/', num,']')
        max_all = run_fea(df.iloc[i,:])

        max_all_list.append(max_all)

    df['max_uz'] = max_all_list

    output_csv_file = f'../data/fea_output_{seed}.csv'
    df.to_csv(output_csv_file, index=False)