
import os
import argparse
import joblib
import numpy as np
import tensorflow as tf
import yaml
import random
from preprocess import prepare_dataset
from soja_model import build_model, train_model, evaluate_model, save_artifacts


# ---------------------------
# Carregar config.yaml
# ---------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Parser mínimo de argumentos
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Arquivo de configuração YAML")
    return p.parse_args()


# ---------------------------
# Função principal
# ---------------------------
def main():
    # ---- 1) Ler argumentos ----
    args = parse_args()

    # ---- 2) Carregar config ----
    cfg = load_config(args.config)

    parquet_path = cfg["parquet"]
    out_dir = cfg["out_dir"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    model_type = cfg["model_type"]
    seed = cfg["seed"]

    # ---- 3) Fixar seeds ----
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Criar diretório de saída
    os.makedirs(out_dir, exist_ok=True)

    print("="*70)
    print("TREINAMENTO DO MODELO DE EXPORTAÇÃO DE SOJA")
    print("="*70)

    print("\n>>> Carregando dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, feature_names, df = prepare_dataset(parquet_path)

    print("\nShapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)
    print("y_test:", y_test.shape)

    # ---------------------------
    # ESCALAMENTO DOS TARGETS
    # ---------------------------
    if y_train.shape[1] != 2:
        raise ValueError(f"O target precisa ter 2 colunas: (Toneladas, Valor_USD_FOB). Encontrado: {y_train.shape[1]}")

    target_names = ["Toneladas", "Valor_USD_FOB"]
    from sklearn.preprocessing import StandardScaler

    y_scalers = {}
    y_train_list = []
    y_val_list = []
    y_test_list = []

    print("\n>>> Escalando as variáveis target...")

    for i, name in enumerate(target_names):
        scaler = StandardScaler()

        col_train = y_train[:, i].reshape(-1, 1)
        col_val = y_val[:, i].reshape(-1, 1)
        col_test = y_test[:, i].reshape(-1, 1)

        scaler.fit(col_train)

        y_train_list.append(scaler.transform(col_train).ravel())
        y_val_list.append(scaler.transform(col_val).ravel())
        y_test_list.append(scaler.transform(col_test).ravel())

        y_scalers[name] = scaler

    y_train_scaled = np.column_stack(y_train_list)
    y_val_scaled = np.column_stack(y_val_list)
    y_test_scaled = np.column_stack(y_test_list)

    # ---------------------------
    # CONSTRUIR MODELO
    # ---------------------------
    print(f"\n>>> Construindo modelo ({model_type})...")
    model = build_model(
        input_dim=X_train.shape[1],
        model_type=model_type,
        lr=lr
    )
    model.summary()

    # ---------------------------
    # TREINAR MODELO
    # ---------------------------
    print("\nIniciando treinamento...")
    hist, ckpt = train_model(
        model,
        X_train, y_train_scaled,
        X_val, y_val_scaled,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size
    )

    print("\nModelo salvo em:", ckpt)

    # ---------------------------
    # SALVAR ARTEFATOS
    # ---------------------------
    model_path, scaler_X_path = save_artifacts(model, scaler_X, y_scalers, out_dir)

    joblib.dump({
        "X_test": X_test,
        "y_test": y_test_scaled,
        "y_test_original": y_test
    }, os.path.join(out_dir, "test_data.joblib"))

    # ---------------------------
    # AVALIAÇÃO
    # ---------------------------
    print("\n>>> Avaliando no conjunto de teste...")
    results = evaluate_model(model, X_test, y_test_scaled, y_scalers=y_scalers)

    print("\nResultados:")
    for name in target_names:
        print(f"\n--- {name} ---")
        print("MAE:", results[f"{name}_mae"])
        print("RMSE:", results[f"{name}_rmse"])

    # ---------------------------
    # SALVAR MÉTRICAS
    # ---------------------------
    joblib.dump(results, os.path.join(out_dir, "metrics.joblib"))

    print("\nTreinamento concluído!")
    print("="*70)


if __name__ == "__main__":
    main()
