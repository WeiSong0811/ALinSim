import pandas as pd
import numpy as np

for seed in [40,41,42,43,44,45,46,47,48,49]:
    csv_file = f'../data/fea_test_{seed}.csv'
    df = pd.read_csv(csv_file)

    num = df.shape[0]

    max_all_list = []
    for i in range(num):
        u_dl_path = f'./output_fea/direct_input_from_csv/seed_{seed}/slab_{i:03d}/u_global_DL.csv'
        u_ll_path = f'./output_fea/direct_input_from_csv/seed_{seed}/slab_{i:03d}/u_global_LL.csv'
        u_llred_path = f'./output_fea/direct_input_from_csv/seed_{seed}/slab_{i:03d}/u_global_LLRED.csv'
        u_sdl_path = f'./output_fea/direct_input_from_csv/seed_{seed}/slab_{i:03d}/u_global_SDL.csv'
        u_dl = pd.read_csv(u_dl_path, header=None).iloc[:, 0].to_numpy(float)
        u_ll = pd.read_csv(u_ll_path, header=None).iloc[:, 0].to_numpy(float)
        u_llred = pd.read_csv(u_llred_path, header=None).iloc[:, 0].to_numpy(float)
        u_sdl = pd.read_csv(u_sdl_path, header=None).iloc[:, 0].to_numpy(float)

        u_dl_uz = u_dl[0::3]
        u_ll_uz = u_ll[0::3]
        u_llred_uz = u_llred[0::3]
        u_sdl_uz = u_sdl[0::3]

        max_dl = np.max(np.abs(u_dl_uz))
        max_ll = np.max(np.abs(u_ll_uz))
        max_llred = np.max(np.abs(u_llred_uz))
        max_sdl = np.max(np.abs(u_sdl_uz))

        max_all = max(max_dl, max_ll, max_llred, max_sdl)

        max_all_list.append(max_all)

    df['max_uz'] = max_all_list

    output_csv_file = f'../data/fea_output_{seed}.csv'
    df.to_csv(output_csv_file, index=False)