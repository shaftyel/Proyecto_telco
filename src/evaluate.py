"""
Script de Evaluacion Avanzada - TelcoVision
Genera metricas adicionales y visualizaciones para el modelo en produccion
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import os
import yaml

# Configurar estilo de graficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_params():
    """Cargar parametros del proyecto"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def load_artifacts():
    """Cargar modelo y datos de prueba"""
    import joblib
    params = load_params()
    
    # Cargar modelo
    model_path = params['paths']['model_path']
    model = joblib.load(model_path)
    
    # Cargar datos procesados
    data_path = params['paths']['processed_data']
    df = pd.read_csv(data_path)
    
    # Separar features y target
    target_col = params['target']
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split train/test
    from sklearn.model_selection import train_test_split
    test_size = params['test_size']
    random_state = params['random_state']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return model, X_test, y_test


def create_plots_directory():
    """Crear directorio de plots si no existe"""
    os.makedirs('plots', exist_ok=True)
    print("[OK] Directorio plots/ creado/verificado")


def plot_confusion_matrix(y_true, y_pred):
    """Generar y guardar matriz de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Matriz de Confusion', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Prediccion', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[OK] Matriz de confusion guardada: plots/confusion_matrix.png")


def plot_roc_curve(y_true, y_proba):
    """Generar y guardar curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Curva ROC - Prediccion de Churn', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Curva ROC guardada: plots/roc_curve.png (AUC: {roc_auc:.4f})")


def plot_precision_recall_curve(y_true, y_proba):
    """Generar y guardar curva Precision-Recall"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='blue', lw=2.5,
             label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('plots/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Curva PR guardada: plots/precision_recall_curve.png (AP: {avg_precision:.4f})")


def plot_feature_importance(model, feature_names):
    """Generar y guardar importancia de features (si el modelo lo soporta)"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Top 15 features
            top_n = min(15, len(feature_names))
            top_indices = indices[:top_n]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(top_n), importances[top_indices], color='steelblue')
            plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
            plt.xlabel('Importancia', fontsize=12)
            plt.title(f'Top {top_n} Features Mas Importantes', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Importancia de features guardada: plots/feature_importance.png")
        else:
            print("[INFO] Modelo no soporta feature_importances_")
    except Exception as e:
        print(f"[WARNING] No se pudo generar feature importance: {e}")


def generate_classification_report(y_true, y_pred):
    """Generar y guardar reporte de clasificacion detallado"""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Guardar en JSON
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Imprimir en consola
    print("\n" + "="*80)
    print("REPORTE DE CLASIFICACION DETALLADO")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))
    print("="*80)
    
    print("[OK] Reporte guardado: metrics/classification_report.json")


def generate_evaluation_summary(y_true, y_pred, y_proba):
    """Generar resumen ejecutivo de la evaluacion"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    summary = {
        "modelo": "Random Forest Conservador (Experimento 5)",
        "dataset": {
            "total_samples": len(y_true),
            "churn_cases": int(y_true.sum()),
            "no_churn_cases": int((y_true == 0).sum()),
            "churn_percentage": float(y_true.mean() * 100)
        },
        "metricas_principales": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_proba))
        },
        "matriz_confusion": {
            "true_negatives": int(confusion_matrix(y_true, y_pred)[0, 0]),
            "false_positives": int(confusion_matrix(y_true, y_pred)[0, 1]),
            "false_negatives": int(confusion_matrix(y_true, y_pred)[1, 0]),
            "true_positives": int(confusion_matrix(y_true, y_pred)[1, 1])
        }
    }
    
    # Guardar resumen
    with open('metrics/evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("[OK] Resumen de evaluacion guardado: metrics/evaluation_summary.json")
    
    return summary


def main():
    """Funcion principal de evaluacion"""
    print("="*80)
    print("EVALUACION AVANZADA DEL MODELO - TELCOVISION")
    print("="*80)
    print()
    
    # Crear directorios
    create_plots_directory()
    os.makedirs('metrics', exist_ok=True)
    
    # Cargar artefactos
    print("\n[INFO] Cargando modelo y datos...")
    model, X_test, y_test = load_artifacts()
    print(f"[OK] Modelo cargado exitosamente")
    print(f"[OK] Datos de prueba: {len(y_test)} muestras")
    
    # Generar predicciones
    print("\n[INFO] Generando predicciones...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("[OK] Predicciones generadas")
    
    # Generar visualizaciones
    print("\n[INFO] Generando visualizaciones...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_feature_importance(model, X_test.columns.tolist())
    
    # Generar reportes
    print("\n[INFO] Generando reportes...")
    generate_classification_report(y_test, y_pred)
    summary = generate_evaluation_summary(y_test, y_pred, y_proba)
    
    # Resumen final
    print("\n" + "="*80)
    print("EVALUACION COMPLETADA EXITOSAMENTE")
    print("="*80)
    print(f"\nMetricas Principales:")
    print(f"   Accuracy:  {summary['metricas_principales']['accuracy']:.4f}")
    print(f"   Precision: {summary['metricas_principales']['precision']:.4f}")
    print(f"   Recall:    {summary['metricas_principales']['recall']:.4f}")
    print(f"   F1-Score:  {summary['metricas_principales']['f1_score']:.4f}")
    print(f"   ROC-AUC:   {summary['metricas_principales']['roc_auc']:.4f}")
    print(f"\nArtefactos generados:")
    print(f"   plots/confusion_matrix.png")
    print(f"   plots/roc_curve.png")
    print(f"   plots/precision_recall_curve.png")
    print(f"   plots/feature_importance.png")
    print(f"   metrics/classification_report.json")
    print(f"   metrics/evaluation_summary.json")
    print("="*80)


if __name__ == "__main__":
    main()