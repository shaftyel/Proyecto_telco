"""
make_data.py

Contexto: TelcoVision - pipeline reproducible para predecir churn.

Este script prepara el dataset `data/raw/telco_churn.csv` para modelado:
- Lee el CSV crudo
- Normaliza nombres de columnas
- Convierte `total_charges` a numérico y rellena faltantes
- Reemplaza cadenas tipo 'No phone service' por 'No'
- Codifica variables categóricas (dummies)
- Guarda el CSV procesado en `data/processed/telco_churn_processed.csv` (por defecto)

Uso (desde la raíz del proyecto):
python src/make_data.py --out data/processed/telco_churn_processed.csv

Entrada: CSV con columna `churn` (0/1), columnas demográficas y de servicio.
Salida: CSV listo para entrenamiento con columnas numéricas y dummies.
"""

import argparse
from pathlib import Path
import pandas as pd


def process_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y transforma el DataFrame del dataset Telco Churn.

    - Convierte `total_charges` a numérico (coerce)
    - Rellena faltantes numéricos con la mediana
    - Elimina `customer_id` si existe
    - Convierte las categóricas a dummies (drop_first=True)
    - Asegura que la columna objetivo se llame `churn` y sea 0/1
    """
    df = df.copy()

    # normalizar nombres de columnas: quitar espacios y pasar a minúsculas
    df.columns = [c.strip().lower() for c in df.columns]

    # eliminar id de cliente si existe
    for id_col in ("customer_id", "customerid", "customer_id "):
        if id_col in df.columns:
            df.drop(columns=[id_col], inplace=True)
            break

    # convertir total_charges a numérico y manejar valores vacíos/espacios
    if "total_charges" in df.columns:
        df["total_charges"] = df["total_charges"].replace("", pd.NA)
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        median_tc = df["total_charges"].median()
        df["total_charges"].fillna(median_tc, inplace=True)

    # columnas numéricas esperadas
    numeric_cols = [c for c in ["age", "tenure_months", "monthly_charges", "total_charges"] if c in df.columns]
    # rellenar numéricos faltantes con mediana
    for c in numeric_cols:
        if df[c].isna().any():
            df[c].fillna(int(df[c].median()), inplace=True)

    # target
    if "churn" in df.columns:
        # asegurar int 0/1
        df["churn"] = pd.to_numeric(df["churn"], errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError("La columna 'churn' no está presente en el dataset")

    # reemplazar valores de servicio que indican ausencia en columnas categóricas
    replace_no_service = ["No phone service", "No internet service", "No phone service ", "No internet service "]
    replace_map = {v: "No" for v in replace_no_service}
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(replace_map)

    # columnas categóricas (excluir la columna objetivo)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != "churn"]

    # crear dummies (incluir NA como categoría si existen)
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

    # crear carpeta destino si no existe
    out.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(out, index=False)

    print(f"Dataset procesado guardado en: {out} (shape={df_processed.shape})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Procesa el CSV de Telco Churn y genera un CSV listo para modelado")
    ap.add_argument("--input", required=False, default="data/raw/telco_churn.csv", help="Ruta al CSV crudo")
    ap.add_argument("--out", required=True, help="Ruta de salida para el CSV procesado")
    args = ap.parse_args()
    main(args.input, args.out)