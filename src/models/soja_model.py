
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, models

print(">>> INÍCIO DO SCRIPT soja_model.py")


# --- Positional Embedding Layer ---
class PositionalEmbedding(layers.Layer):
    """Adiciona positional encoding aprendível aos tokens."""
    
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(self.seq_len, self.d_model),
            initializer="random_normal",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model
        })
        return config


def _transformer_encoder_block(x, d_model, head_size=32, num_heads=4, ff_dim=128, dropout=0.1):
    """Bloco encoder do Transformer com LayerNorm e skip connections."""
    
    # Multi-head self-attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout
    )(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed-forward network
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    
    return x


def build_model(input_dim: int,
                model_type: str = "mlp",
                d_model: int = 128,
                num_blocks: int = 3,
                head_size: int = 64,
                num_heads: int = 8,
                ff_dim: int = 128,
                dropout: float = 0.2,
                hidden_units=[512, 256, 128, 64],
                lr: float = 3e-4) -> keras.Model:
    """
    Constrói modelo para previsão de exportação de soja.
    
    Args:
        input_dim: Número de features de entrada
        model_type: "transformer" ou "mlp"
        d_model: Dimensão dos embeddings no transformer
        num_blocks: Número de blocos transformer
        head_size: Dimensão de cada attention head
        num_heads: Número de attention heads
        ff_dim: Dimensão da feed-forward network
        dropout: Taxa de dropout
        hidden_units: Unidades das camadas densas (apenas para MLP)
        lr: Learning rate
    
    Returns:
        Modelo Keras compilado
    """
    
    inp = layers.Input(shape=(input_dim,), name="input")

    # ============================================================
    # MLP BASELINE
    # ============================================================
    if model_type.lower() == "mlp":
        x = inp
        for i, units in enumerate(hidden_units):
            x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
        backbone = x

    # ============================================================
    # TRANSFORMER TABULAR
    # ============================================================
    else:
        # Cada feature vira um token de dimensão 1
        x = layers.Reshape((input_dim, 1))(inp)
        
        # Projeta cada token para d_model dimensões
        x = layers.Dense(d_model, name="token_projection")(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Adiciona positional encoding
        x = PositionalEmbedding(seq_len=input_dim, d_model=d_model)(x)
        
        # Stack de blocos Transformer
        for i in range(num_blocks):
            x = _transformer_encoder_block(
                x,
                d_model=d_model,
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
        
        # Agrega informação dos tokens
        backbone = layers.GlobalAveragePooling1D(name="pooling")(x)
        backbone = layers.Dropout(dropout)(backbone)

    # ============================================================
    # HEADS DE REGRESSÃO
    # ============================================================
    out_ton = layers.Dense(1, name="Toneladas")(backbone)
    out_val = layers.Dense(1, name="Valor_USD_FOB")(backbone)

    model = models.Model(
        inputs=inp,
        outputs=[out_ton, out_val],
        name=f"soja_model_{model_type}"
    )

    # Compilação com AdamW
    opt = optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
    
    model.compile(
        optimizer=opt,
        loss={"Toneladas": "mse", "Valor_USD_FOB": "mse"},
        metrics={"Toneladas": ["mae"], "Valor_USD_FOB": ["mae"]}
    )

    return model


def train_model(model: keras.Model,
                X_train: np.ndarray,
                y_train_scaled: np.ndarray,
                X_val: np.ndarray,
                y_val_scaled: np.ndarray,
                out_dir: str,
                epochs: int = 100,
                batch_size: int = 64,
                patience: int = 10,
                reduce_lr: bool = True) -> Tuple[keras.callbacks.History, str]:
    """
    Treina o modelo com early stopping e checkpoint.
    
    Args:
        model: Modelo Keras
        X_train, X_val: Features de treino e validação
        y_train_scaled, y_val_scaled: Targets escalados (shape: [n, 2])
        out_dir: Diretório para salvar checkpoints
        epochs: Número máximo de épocas
        batch_size: Tamanho do batch
        patience: Paciência para early stopping
        reduce_lr: Se True, reduz LR quando val_loss platô
    
    Returns:
        Tupla (history, checkpoint_path)
    """
    
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "soja_model_best.keras")

    # ============================================================
    # VALIDAÇÃO E PREPARAÇÃO DOS DADOS
    # ============================================================
    print("\n======== DEBUG DADOS TREINO =========")
    print(f"X_train.shape = {X_train.shape}")
    print(f"y_train_scaled.shape = {y_train_scaled.shape}")

    # Garantir formato correto
    y_train = np.asarray(y_train_scaled)
    y_val = np.asarray(y_val_scaled)
    
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)
    
    # Pegar apenas as 2 primeiras colunas (Toneladas e Valor_USD_FOB)
    if y_train.shape[1] > 2:
        print(f"⚠️  Aviso: y_train tem {y_train.shape[1]} colunas. Usando apenas as 2 primeiras.")
        y_train = y_train[:, :2]
    if y_val.shape[1] > 2:
        print(f"⚠️  Aviso: y_val tem {y_val.shape[1]} colunas. Usando apenas as 2 primeiras.")
        y_val = y_val[:, :2]
    
    # Validar que temos pelo menos 2 colunas
    if y_train.shape[1] < 2:
        raise ValueError(f"y_train deve ter pelo menos 2 colunas, mas tem {y_train.shape[1]}")
    if y_val.shape[1] < 2:
        raise ValueError(f"y_val deve ter pelo menos 2 colunas, mas tem {y_val.shape[1]}")

    # Debug estatístico
    print("\nPrimeiras 5 linhas y_train_scaled:")
    print(y_train[:5])
    
    import pandas as pd
    df_debug = pd.DataFrame(y_train, columns=["Toneladas", "Valor_USD_FOB"])
    print("\nEstatísticas dos targets (escalados):")
    print(df_debug.describe())

    # Estrutura esperada pelo Keras (lista de arrays)
    y_train_list = [y_train[:, 0], y_train[:, 1]]
    y_val_list = [y_val[:, 0], y_val[:, 1]]

    # ============================================================
    # CALLBACKS
    # ============================================================
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    if reduce_lr:
        cbs.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, patience // 3),
                min_lr=1e-7,
                verbose=1
            )
        )

    # ============================================================
    # TREINAMENTO
    # ============================================================
    print(f"\n>>> Iniciando treinamento por até {epochs} épocas...")
    
    hist = model.fit(
        X_train,
        y_train_list,
        validation_data=(X_val, y_val_list),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=2
    )

    # Salvar histórico
    hist_path = os.path.join(out_dir, "history.joblib")
    joblib.dump(hist.history, hist_path)
    print(f"\n>>> Histórico salvo em {hist_path}")
    print(f">>> Melhor modelo salvo em {ckpt_path}")

    return hist, ckpt_path


def evaluate_model(model: keras.Model,
                   X_test: np.ndarray,
                   y_test_scaled: np.ndarray,
                   y_scalers: Optional[Dict[str, object]] = None) -> dict:
    """
    Avalia o modelo no conjunto de teste.
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test_scaled: Targets escalados
        y_scalers: Dict com scalers para Toneladas e Valor_USD_FOB
    
    Returns:
        Dict com métricas MSE e MAE para cada target
    """
    
    preds = model.predict(X_test, verbose=0)
    
    # Converter predições para array numpy
    if isinstance(preds, list):
        preds_arr = np.column_stack(preds)
    else:
        preds_arr = np.asarray(preds)
    
    # Garantir formato correto dos targets
    y_arr = np.asarray(y_test_scaled)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    
    # Pegar apenas as 2 primeiras colunas se houver mais
    if y_arr.shape[1] > 2:
        print(f"⚠️  Aviso: y_test tem {y_arr.shape[1]} colunas. Usando apenas as 2 primeiras.")
        y_arr = y_arr[:, :2]
    
    if y_arr.shape[1] < 2:
        raise ValueError(f"y_test deve ter pelo menos 2 colunas, mas tem {y_arr.shape[1]}")

    names = ["Toneladas", "Valor_USD_FOB"]
    results = {}

    for i, name in enumerate(names):
        y_true = y_arr[:, i]
        y_pred = preds_arr[:, i]

        # Inverter escala se disponível
        if y_scalers and name in y_scalers and y_scalers[name] is not None:
            try:
                y_true = y_scalers[name].inverse_transform(
                    y_true.reshape(-1, 1)
                ).ravel()
                y_pred = y_scalers[name].inverse_transform(
                    y_pred.reshape(-1, 1)
                ).ravel()
            except Exception as e:
                print(f"Aviso: Não foi possível inverter escala para {name}: {e}")

        # Calcular métricas
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        
        results[f"{name}_mse"] = mse
        results[f"{name}_rmse"] = rmse
        results[f"{name}_mae"] = mae

    # Métricas globais
    results["total_mse"] = float(np.mean((y_arr - preds_arr) ** 2))
    results["total_mae"] = float(np.mean(np.abs(y_arr - preds_arr)))
    
    return results


def save_artifacts(model: keras.Model,
                   scaler_X,
                   y_scalers: dict,
                   out_dir: str,
                   model_name: str = "soja_model") -> Tuple[str, str]:
    """
    Salva modelo e scalers.
    
    Args:
        model: Modelo treinado
        scaler_X: Scaler das features
        y_scalers: Dict com scalers dos targets
        out_dir: Diretório de saída
        model_name: Nome base do arquivo do modelo
    
    Returns:
        Tupla (model_path, scaler_X_path)
    """
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Salvar modelo
    model_path = os.path.join(out_dir, f"{model_name}.keras")
    try:
        model.save(model_path)
        print(f">>> Modelo salvo em {model_path}")
    except Exception as e:
        print(f"Erro ao salvar .keras, tentando .h5: {e}")
        model_path = os.path.join(out_dir, f"{model_name}.h5")
        model.save(model_path)
        print(f">>> Modelo salvo em {model_path}")
    
    # Salvar scalers
    scaler_X_path = os.path.join(out_dir, "scaler_X.joblib")
    scaler_y_path = os.path.join(out_dir, "scaler_y.joblib")
    
    if scaler_X is not None:
        joblib.dump(scaler_X, scaler_X_path)
        print(f">>> Scaler X salvo em {scaler_X_path}")
    
    if y_scalers is not None:
        joblib.dump(y_scalers, scaler_y_path)
        print(f">>> Scalers Y salvos em {scaler_y_path}")

    return model_path, scaler_X_path


print(">>> FIM DO SCRIPT soja_model.py")