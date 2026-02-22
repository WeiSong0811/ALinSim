import pandas as pd

def check_duplicate_rows(csv_path):
    df = pd.read_csv(csv_path)

    # 是否存在重复行
    has_duplicates = df.duplicated().any()
    print("是否存在重复行:", has_duplicates)

    if has_duplicates:
        dup_rows = df[df.duplicated(keep=False)]
        print("\n重复的行如下：")
        print(dup_rows)

        print("\n重复行数量:", dup_rows.shape[0])

check_duplicate_rows("FEA_inpute_pos.csv")
