# predict_soja.py
import joblib
import numpy as np
from tensorflow.keras.models import load_model

scaler_X = joblib.load("artifacts/scaler_X.joblib")
y_scalers = joblib.load("artifacts/scaler_y.joblib")
model = load_model("artifacts/soja_model.keras")

X_new = np.zeros((1, scaler_X.n_features_in_))  # exemplo zeros
X_new_scaled = scaler_X.transform(X_new)

preds = model.predict(X_new_scaled)

ton = preds[0].ravel()
kg = preds[1].ravel()
val = preds[2].ravel()

ton_r = y_scalers["Toneladas"].inverse_transform(ton.reshape(-1,1)).ravel()

val_r = y_scalers["Valor_USD_FOB"].inverse_transform(val.reshape(-1,1)).ravel()

print("Predição (unscaled):")
print({"Toneladas": ton_r[0], "Valor_USD_FOB": val_r[0]})
