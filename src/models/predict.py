# predict_soja.py
import joblib
import numpy as np
from tensorflow.keras.models import load_model

scaler_X = joblib.load("artifacts/scaler_X.joblib")
y_scalers = joblib.load("artifacts/scaler_y.joblib")
model = load_model("artifacts/soja_model.keras")

# exemplo: X_new deve ter shape (n_samples, n_features)
# substitua pelos 137 valores do seu feature_names
X_new = np.zeros((1, scaler_X.n_features_in_))  # exemplo zeros
X_new_scaled = scaler_X.transform(X_new)

preds = model.predict(X_new_scaled)
# preds is list of arrays [ton, kg, val] in our build order
ton = preds[0].ravel()
kg = preds[1].ravel()
val = preds[2].ravel()

# inverse scale per target
ton_r = y_scalers["Toneladas"].inverse_transform(ton.reshape(-1,1)).ravel()
kg_r = y_scalers["Quilograma_Liquido"].inverse_transform(kg.reshape(-1,1)).ravel()
val_r = y_scalers["Valor_USD_FOB"].inverse_transform(val.reshape(-1,1)).ravel()

print("Predição (unscaled):")
print({"Toneladas": ton_r[0], "Quilograma_Liquido": kg_r[0], "Valor_USD_FOB": val_r[0]})
