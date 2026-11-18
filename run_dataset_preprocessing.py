import os
import sys
import random
import pandas as pd
import numpy as np
from sklearn.utils import resample

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

BASE_PATH = r"C:\Study\HocKy6\MH&MM\qml-ddos"
DATASET_DIR = os.path.join(BASE_PATH, "dataset")

RANDOM_SEED = 1234
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_TRAIN = 80000
N_TEST = 20000

N_TRAIN_SMALL = 800
N_TEST_SMALL = 200

N_TRAIN_TINY = 80
N_TEST_TINY = 20

SYN_0_FRACTION = 0.9
SYN_1_FRACTION = 1 - SYN_0_FRACTION

def main():
    print("==================================================")
    print("BAT DAU TIEN XU LY VA PHAN TANG DU LIEU CICDDoS2019")
    print("Theo dung quy trinh dev/00_dataset_import.ipynb")
    print("==================================================")

    p_tr = os.path.join(DATASET_DIR, "CICDDoS2019", "CSVs", "01-12", "Syn.csv")
    p_test = os.path.join(DATASET_DIR, "CICDDoS2019", "CSVs", "03-11", "Syn.csv")

    print(f"\n[1/5] Dang doc file Train: {p_tr} ...")
    df_tr_day_syn = pd.read_csv(p_tr)
    print(f"      Train shape goc: {df_tr_day_syn.shape}")

    print(f"\n[2/5] Dang doc file Test: {p_test} ...")
    df_test_day_syn = pd.read_csv(p_test)
    print(f"      Test shape goc: {df_test_day_syn.shape}")

    print("\n[3/5] Loc bo cac cot metadata va lam sach...")
    columns_to_remove = ["Unnamed: 0", "Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port", " Timestamp", "SimillarHTTP"]
    
    # Strip any potential leading/trailing whitespace in column names
    col_map_tr = {c: c.strip() for c in df_tr_day_syn.columns}
    col_map_test = {c: c.strip() for c in df_test_day_syn.columns}
    
    df_tr_day_syn.rename(columns=col_map_tr, inplace=True)
    df_test_day_syn.rename(columns=col_map_test, inplace=True)
    
    clean_cols_to_remove = [c.strip() for c in columns_to_remove]
    existing_tr_cols = [c for c in clean_cols_to_remove if c in df_tr_day_syn.columns]
    existing_test_cols = [c for c in clean_cols_to_remove if c in df_test_day_syn.columns]
    
    df_tr_day_syn.drop(columns=existing_tr_cols, inplace=True)
    df_test_day_syn.drop(columns=existing_test_cols, inplace=True)

    df_tr_day_syn.dropna(inplace=True)
    df_test_day_syn.dropna(inplace=True)

    df_tr_day_syn.drop_duplicates(inplace=True)
    df_test_day_syn.drop_duplicates(inplace=True)

    # Encode Label
    label_col = "Label" if "Label" in df_tr_day_syn.columns else " Label"
    df_tr_day_syn.replace(to_replace={label_col: "Syn"}, value=0, inplace=True)
    df_tr_day_syn.replace(to_replace={label_col: "BENIGN"}, value=1, inplace=True)

    df_test_day_syn.replace(to_replace={label_col: "Syn"}, value=0, inplace=True)
    df_test_day_syn.replace(to_replace={label_col: "BENIGN"}, value=1, inplace=True)

    # Filter non-finite
    df_tr_day_syn = df_tr_day_syn[np.isfinite(df_tr_day_syn).all(1)]
    df_test_day_syn = df_test_day_syn[np.isfinite(df_test_day_syn).all(1)]

    print(f"      Train sach: {df_tr_day_syn.shape} | Test sach: {df_test_day_syn.shape}")

    print("\n[4/5] Phan tang va can bang ti le 90% Syn (0) : 10% Benign (1)...")
    df_tr_day_syn_0 = df_tr_day_syn[df_tr_day_syn[label_col] == 0]
    df_tr_day_syn_1 = df_tr_day_syn[df_tr_day_syn[label_col] == 1]

    df_test_day_syn_0 = df_test_day_syn[df_test_day_syn[label_col] == 0]
    df_test_day_syn_1 = df_test_day_syn[df_test_day_syn[label_col] == 1]

    # Full
    df_tr_0_full = resample(df_tr_day_syn_0, replace=True, n_samples=round(N_TRAIN * SYN_0_FRACTION), random_state=RANDOM_SEED)
    df_tr_1_full = resample(df_tr_day_syn_1, replace=True, n_samples=round(N_TRAIN * SYN_1_FRACTION), random_state=RANDOM_SEED)
    df_te_0_full = resample(df_test_day_syn_0, replace=True, n_samples=round(N_TEST * SYN_0_FRACTION), random_state=RANDOM_SEED)
    df_te_1_full = resample(df_test_day_syn_1, replace=True, n_samples=round(N_TEST * SYN_1_FRACTION), random_state=RANDOM_SEED)

    # Small
    df_tr_0_small = resample(df_tr_day_syn_0, replace=True, n_samples=round(N_TRAIN_SMALL * SYN_0_FRACTION), random_state=RANDOM_SEED)
    df_tr_1_small = resample(df_tr_day_syn_1, replace=True, n_samples=round(N_TRAIN_SMALL * SYN_1_FRACTION), random_state=RANDOM_SEED)
    df_te_0_small = resample(df_test_day_syn_0, replace=True, n_samples=round(N_TEST_SMALL * SYN_0_FRACTION), random_state=RANDOM_SEED)
    df_te_1_small = resample(df_test_day_syn_1, replace=True, n_samples=round(N_TEST_SMALL * SYN_1_FRACTION), random_state=RANDOM_SEED)

    # Tiny
    df_tr_0_tiny = resample(df_tr_day_syn_0, replace=True, n_samples=round(N_TRAIN_TINY * SYN_0_FRACTION), random_state=RANDOM_SEED)
    df_tr_1_tiny = resample(df_tr_day_syn_1, replace=True, n_samples=round(N_TRAIN_TINY * SYN_1_FRACTION), random_state=RANDOM_SEED)
    df_te_0_tiny = resample(df_test_day_syn_0, replace=True, n_samples=round(N_TEST_TINY * SYN_0_FRACTION), random_state=RANDOM_SEED)
    df_te_1_tiny = resample(df_test_day_syn_1, replace=True, n_samples=round(N_TEST_TINY * SYN_1_FRACTION), random_state=RANDOM_SEED)

    # Concat
    df_tr_full = pd.concat([df_tr_0_full, df_tr_1_full])
    df_te_full = pd.concat([df_te_0_full, df_te_1_full])

    df_tr_small = pd.concat([df_tr_0_small, df_tr_1_small])
    df_te_small = pd.concat([df_te_0_small, df_te_1_small])

    df_tr_tiny = pd.concat([df_tr_0_tiny, df_tr_1_tiny])
    df_te_tiny = pd.concat([df_te_0_tiny, df_te_1_tiny])

    print("\n[5/5] Dang luu cac tap du lieu vao dataset/ ...")
    out_tr_tiny = os.path.join(DATASET_DIR, "tr_day_syn_tiny.csv")
    out_te_tiny = os.path.join(DATASET_DIR, "test_day_syn_tiny.csv")
    out_tr_small = os.path.join(DATASET_DIR, "tr_day_syn_small.csv")
    out_te_small = os.path.join(DATASET_DIR, "test_day_syn_small.csv")
    out_tr_full = os.path.join(DATASET_DIR, "tr_day_syn.csv")
    out_te_full = os.path.join(DATASET_DIR, "test_day_syn.csv")

    df_tr_tiny.to_csv(out_tr_tiny, index=False)
    df_te_tiny.to_csv(out_te_tiny, index=False)
    print(f"      -> Tiny: {out_tr_tiny} ({df_tr_tiny.shape}) & {out_te_tiny} ({df_te_tiny.shape})")

    df_tr_small.to_csv(out_tr_small, index=False)
    df_te_small.to_csv(out_te_small, index=False)
    print(f"      -> Small: {out_tr_small} ({df_tr_small.shape}) & {out_te_small} ({df_te_small.shape})")

    df_tr_full.to_csv(out_tr_full, index=False)
    df_te_full.to_csv(out_te_full, index=False)
    print(f"      -> Full: {out_tr_full} ({df_tr_full.shape}) & {out_te_full} ({df_te_full.shape})")

    print("\n>>> HOAN TAT TOAN BO TIEN XU LY DU LIEU CHUAN XAC 100%!")

if __name__ == "__main__":
    main()
