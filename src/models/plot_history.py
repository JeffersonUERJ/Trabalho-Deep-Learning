# plot_history.py
import joblib
import matplotlib.pyplot as plt
import numpy as np

h = joblib.load("C:/Trabalho Deep/src/models/artifacts/history.joblib")

# loss
plt.figure(figsize=(8,4))
plt.plot(h["loss"], label="train_loss")
plt.plot(h["val_loss"], label="val_loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
plt.tight_layout(); plt.show()

# per-output MAE (Keras may name metrics differently: check keys)
keys = sorted([k for k in h.keys() if "mae" in k])
for k in keys:
    plt.figure(figsize=(8,3))
    plt.plot(h[k], label=k)
    plt.xlabel("epoch"); plt.ylabel("mae"); plt.legend(); plt.title(k)
    plt.tight_layout(); plt.show()
