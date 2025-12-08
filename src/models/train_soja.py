# train_soja.py (CORRIGIDO)
import os
import argparse
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from preprocess import prepare_dataset
from soja_model import build_model, train_model, evaluate_model, save_artifacts

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_type", default="transformer", choices=["transformer","mlp"])
    return p.parse_args()

def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("="*70)
    print("TREINAMENTO DO MODELO DE EXPORTAÇÃO DE SOJA")
    print("="*70)
    
    print("\n>>> Carregando e preparando dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, feature_names, df = prepare_dataset(args.parquet)

    print(f"\n📊 Shapes dos dados:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val:   {X_val.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_val:   {y_val.shape}")
    print(f"   y_test:  {y_test.shape}")

    # ================================================================
    # ESCALANDO OS TARGETS (APENAS 2 COLUNAS: Toneladas e Valor_USD_FOB)
    # ================================================================
    
    # Validar que y tem exatamente 2 colunas
    if y_train.shape[1] != 2:
        raise ValueError(f"y_train deve ter 2 colunas (Toneladas, Valor_USD_FOB), mas tem {y_train.shape[1]}")
    
    target_names = ["Toneladas", "Valor_USD_FOB"]
    y_scalers = {}
    y_train_scaled_list = []
    y_val_scaled_list = []
    y_test_scaled_list = []
    
    print(f"\n>>> Escalando targets com StandardScaler...")
    
    for i, name in enumerate(target_names):
        print(f"   Processando: {name}")
        
        # Criar scaler
        scaler = StandardScaler()
        
        # Extrair coluna
        col_train = y_train[:, i].reshape(-1, 1)
        col_val = y_val[:, i].reshape(-1, 1)
        col_test = y_test[:, i].reshape(-1, 1)
        
        # Fit no treino e transform em todos
        scaler.fit(col_train)
        y_train_scaled_list.append(scaler.transform(col_train).ravel())
        y_val_scaled_list.append(scaler.transform(col_val).ravel())
        y_test_scaled_list.append(scaler.transform(col_test).ravel())
        
        # Guardar scaler
        y_scalers[name] = scaler
        
        # Mostrar estatísticas
        print(f"      Original - Min: {col_train.min():.2f}, Max: {col_train.max():.2f}, Média: {col_train.mean():.2f}")
        print(f"      Escalado - Min: {scaler.transform(col_train).min():.2f}, Max: {scaler.transform(col_train).max():.2f}")
    
    # Stack em arrays (n, 2)
    y_train_scaled = np.column_stack(y_train_scaled_list)
    y_val_scaled = np.column_stack(y_val_scaled_list)
    y_test_scaled = np.column_stack(y_test_scaled_list)
    
    print(f"\n✅ Shapes finais (escalados):")
    print(f"   y_train_scaled: {y_train_scaled.shape}")
    print(f"   y_val_scaled:   {y_val_scaled.shape}")
    print(f"   y_test_scaled:  {y_test_scaled.shape}")
    
    # ================================================================
    # CONSTRUIR E TREINAR MODELO
    # ================================================================
    
    print(f"\n>>> Construindo modelo: {args.model_type}")
    model = build_model(
        input_dim=X_train.shape[1],
        model_type=args.model_type,
        lr=args.lr
    )
    
    print(f"\n>>> Resumo do modelo:")
    model.summary()
    
    print(f"\n>>> Iniciando treinamento...")
    print(f"   Épocas: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    
    hist, ckpt = train_model(
        model, X_train, y_train_scaled, X_val, y_val_scaled,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"\n✅ Treinamento finalizado!")
    print(f"   Melhor modelo salvo em: {ckpt}")
    
    # ================================================================
    # SALVAR ARTEFATOS
    # ================================================================
    
    print(f"\n>>> Salvando artefatos...")
    model_path, scaler_X_path = save_artifacts(model, scaler_X, y_scalers, args.out_dir)
    print(f"   Modelo: {model_path}")
    print(f"   Scaler X: {scaler_X_path}")
    
    # Salvar dados de teste para avaliação posterior
    test_data_path = os.path.join(args.out_dir, "test_data.joblib")
    joblib.dump({
        "X_test": X_test,
        "y_test": y_test_scaled,
        "y_test_original": y_test  # Guardar também os valores originais
    }, test_data_path)
    print(f"   Dados de teste: {test_data_path}")
    
    # ================================================================
    # AVALIAR NO CONJUNTO DE TESTE (ESCALA ORIGINAL)
    # ================================================================
    
    print("\n" + "="*70)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*70)
    
    results = evaluate_model(model, X_test, y_test_scaled, y_scalers=y_scalers)
    
    print("\n📊 Métricas em ESCALA ORIGINAL:")
    print("-" * 70)
    
    for name in target_names:
        mae = results.get(f"{name}_mae", 0)
        rmse = results.get(f"{name}_rmse", 0)
        
        print(f"\n{name}:")
        print(f"   MAE:  {mae:,.2f}")
        print(f"   RMSE: {rmse:,.2f}")
        
        # Mostrar estatísticas dos valores reais
        col_test = y_test[:, target_names.index(name)]
        print(f"\n   Valores Reais no Teste:")
        print(f"   • Min:    {col_test.min():,.2f}")
        print(f"   • Max:    {col_test.max():,.2f}")
        print(f"   • Média:  {col_test.mean():,.2f}")
        print(f"   • Mediana: {np.median(col_test):,.2f}")
        print(f"   • Std:    {col_test.std():,.2f}")
        
        # Erro percentual médio
        mape = (mae / col_test.mean()) * 100
        print(f"\n   Erro Relativo (MAPE): {mape:.2f}%")
    
    print("\n" + "-" * 70)
    print(f"Total MSE:  {results['total_mse']:.4f}")
    print(f"Total MAE:  {results['total_mae']:.4f}")
    print("="*70)
    
    # ================================================================
    # SALVAR MÉTRICAS E METADADOS
    # ================================================================
    
    metrics_path = os.path.join(args.out_dir, "metrics.joblib")
    joblib.dump(results, metrics_path)
    print(f"\n✅ Métricas salvas em: {metrics_path}")
    
    meta_path = os.path.join(args.out_dir, "meta.joblib")
    joblib.dump({
        "feature_names": feature_names,
        "target_names": target_names,
        "model_type": args.model_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr
    }, meta_path)
    print(f"✅ Metadados salvos em: {meta_path}")
    
    print("\n" + "="*70)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*70)
    
    # Mostrar próximos passos
    print("\n📝 PRÓXIMOS PASSOS:")
    print(f"   1. Ver gráficos detalhados:")
    print(f"      python evaluate_soja.py --test_data {test_data_path}")
    print(f"\n   2. Fazer predições em novos dados:")
    print(f"      python predict_soja.py --model {model_path} --data <novo_arquivo.csv>")
    print("\n")

if __name__ == "__main__":
    main()