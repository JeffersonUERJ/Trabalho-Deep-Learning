# predict_soja.py
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from soja_model import PositionalEmbedding   


# carregamento dos scalers
scaler_X = joblib.load("artifacts/scaler_X.joblib")
y_scalers = joblib.load("artifacts/scaler_y.joblib")

# carregamento do modelo com custom_objects
model = load_model(
    "artifacts/soja_model.keras",
    custom_objects={"PositionalEmbedding": PositionalEmbedding}
)

# exemplo de entrada
X_new = np.zeros((1, scaler_X.n_features_in_))
X_new_scaled = scaler_X.transform(X_new)

# predição
preds = model.predict(X_new_scaled)

ton = preds[0].ravel()
#kg = preds[1].ravel()
val = preds[1].ravel()

# inversões
ton_r = y_scalers["Toneladas"].inverse_transform(ton.reshape(-1, 1)).ravel()
val_r = y_scalers["Valor_USD_FOB"].inverse_transform(val.reshape(-1, 1)).ravel()

print("Predição (unscaled):")
print({"Toneladas": ton_r[0], "Valor_USD_FOB": val_r[0]})
