
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)

nama_kelas = [
    "Pertanian", "Pertambangan", "Penebangan Kayu",
    "Pembangunan Jalan", "Gangguan Alam"
]

# Membuat citra simulasi
ukuran = 32

def buat_citra(kelas):
    img = np.random.rand(ukuran, ukuran, 3) * 0.2

    if kelas == 0:
        img[:, 5::7, 1] += 0.7
    elif kelas == 1:
        img[10:23, 10:23, 0] += 0.7
    elif kelas == 2:
        img[5::6, :, 0] += 0.7
    elif kelas == 3:
        img[:, 14:18, :] += 0.7
    else:
        mask = np.random.rand(ukuran, ukuran) > 0.7
        img[mask, 0] += 0.7

    return np.clip(img, 0, 1).astype("float32")

X, y = [], []

for kelas in range(5):
    for _ in range(200):
        X.append(buat_citra(kelas))
        y.append(kelas)

X = np.array(X)
y = np.array(y)

print("Jumlah citra:", len(X))
print("Ukuran citra:", X.shape[1:])

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Data pelatihan:", len(X_train))
print("Data pengujian:", len(X_test))

# Membuat model CNN
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Melatih model
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(f"\nAkurasi pengujian: {accuracy:.4f}")
print("\nLaporan klasifikasi:")
print(classification_report(
    y_test, y_pred,
    labels=np.arange(5),
    target_names=nama_kelas,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Grafik akurasi dan loss
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(model.history.history["accuracy"], label="Training")
plt.plot(model.history.history["val_accuracy"], label="Validation")
plt.title("Akurasi Model CNN")
plt.xlabel("Epoch")
plt.ylabel("Akurasi")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(model.history.history["loss"], label="Training")
plt.plot(model.history.history["val_loss"], label="Validation")
plt.title("Loss Model CNN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

print("\nKesimpulan: CNN digunakan untuk klasifikasi lima kategori")
print("penyebab perubahan hutan berdasarkan citra simulasi.")
print("Model ini belum menggunakan citra satelit asli.")
