"""
data_prep.py

Script de limpieza y transformación de datos para TelcoVision.
- Lee el dataset crudo `data/raw/telco_churn.csv`
- Aplica limpieza, conversión de tipos, codificación de categóricas y genera el dataset limpio
- Guarda el resultado en `data/processed/telco_churn_processed.csv`

Uso:
python src/data_prep.py --input data/raw/telco_churn.csv --out data/processed/telco_churn_processed.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def process_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for id_col in ("customer_id", "customerid", "customer_id "):
        if id_col in df.columns:
            df.drop(columns=[id_col], inplace=True)
            break
    if "total_charges" in df.columns:
        df["total_charges"] = df["total_charges"].replace("", pd.NA)
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        median_tc = df["total_charges"].median()
        df["total_charges"].fillna(median_tc, inplace=True)
    numeric_cols = [c for c in ["age", "tenure_months", "monthly_charges", "total_charges"] if c in df.columns]
    for c in numeric_cols:
        if df[c].isna().any():
            df[c].fillna(int(df[c].median()), inplace=True)
    if "churn" in df.columns:
        df["churn"] = pd.to_numeric(df["churn"], errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError("La columna 'churn' no está presente en el dataset")
    replace_no_service = ["No phone service", "No internet service", "No phone service ", "No internet service "]
    replace_map = {v: "No" for v in replace_no_service}
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(replace_map)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "churn"]
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols], drop_first=True, dummy_na=True)
        df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    return df

def main(input_path: str, out_path: str):
    inp = Path(input_path)
    out = Path(out_path)
    if not inp.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {inp}")
    df = pd.read_csv(inp)
    df_processed = process_telco(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(out, index=False)
    print(f"Dataset limpio guardado en: {out} (shape={df_processed.shape})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Ruta al CSV crudo")
    ap.add_argument("--out", required=True, help="Ruta al CSV limpio")
    args = ap.parse_args()
    main(args.input, args.out)
