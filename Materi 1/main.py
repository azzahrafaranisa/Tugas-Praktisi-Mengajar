# STUDI KASUS: PT MAJU BERSAMA
# Analisis Penjualan dan Prediksi Keterlambatan Pembayaran

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)



# PERSIAPAN FOLDER OUTPUT
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. MEMBUAT DATA SIMULASI

np.random.seed(42)
n = 600

data = pd.DataFrame({
    "Bulan": np.random.choice(
        ["Januari", "Februari", "Maret",
         "April", "Mei", "Juni"], n
    ),
    "Pelanggan": [f"P{i+1}" for i in range(n)],
    "Jenis_Pelanggan": np.random.choice(
        ["Retail", "Grosir", "Bisnis"], n
    ),
    "Penjualan": np.random.randint(
        500_000, 5_000_000, n
    ),
    "Biaya_Operasional": np.random.randint(
        200_000, 3_500_000, n
    ),
    "Diskon": np.random.randint(0, 31, n),
    "Riwayat_Terlambat": np.random.choice(
        [0, 1, 2, 3], n,
        p=[0.55, 0.25, 0.15, 0.05]
    ),
    "Lama_Berlangganan": np.random.randint(1, 61, n)
})

probabilitas = (
    -2.0
    + 0.65 * data["Riwayat_Terlambat"]
    + 0.025 * data["Diskon"]
    - 0.015 * data["Lama_Berlangganan"]
)

probabilitas = 1 / (1 + np.exp(-probabilitas))

data["Terlambat_Bayar"] = (
    np.random.random(n) < probabilitas
).astype(int)

data.loc[5, "Diskon"] = np.nan
data.loc[10, "Jenis_Pelanggan"] = np.nan


# PEMERIKSAAN DAN PEMBERSIHAN DATA
print("\n" + "=" * 60)
print("1. PEMERIKSAAN DATA")
print("=" * 60)

print("\nLima data pertama:")
print(data.head().to_string(index=False))

print("\nUkuran data:", data.shape)
print("\nJumlah nilai kosong:")
print(data.isnull().sum())

print("\nJumlah data duplikat:", data.duplicated().sum())

data = data.drop_duplicates().copy()

data["Diskon"] = data["Diskon"].fillna(
    data["Diskon"].median()
)

data["Jenis_Pelanggan"] = data["Jenis_Pelanggan"].fillna(
    data["Jenis_Pelanggan"].mode()[0]
)

data["Keuntungan"] = (
    data["Penjualan"] - data["Biaya_Operasional"]
)

print("\nJumlah nilai kosong setelah dibersihkan:")
print(data.isnull().sum())


data.to_csv(
    OUTPUT_DIR / "data_bersih.csv",
    index=False
)



print("\n" + "=" * 60)
print("2. ANALISIS PENJUALAN DAN KEUNTUNGAN")
print("=" * 60)

ringkasan = data.groupby("Bulan")[
    ["Penjualan", "Biaya_Operasional", "Keuntungan"]
].sum()

urutan_bulan = [
    "Januari", "Februari", "Maret",
    "April", "Mei", "Juni"
]
ringkasan = ringkasan.reindex(urutan_bulan)

print("\nRingkasan per bulan:")
print(ringkasan.to_string())

ringkasan.to_csv(
    OUTPUT_DIR / "ringkasan_bulanan.csv"
)


ax = ringkasan.plot(
    kind="bar",
    figsize=(10, 5)
)

ax.set_title("Penjualan, Biaya, dan Keuntungan per Bulan")
ax.set_xlabel("Bulan")
ax.set_ylabel("Total (Rp)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "analisis_bulanan.png",
    dpi=150
)
plt.show()
plt.close()

# Analisis berdasarkan jenis pelanggan
analisis_pelanggan = data.groupby("Jenis_Pelanggan").agg(
    Total_Penjualan=("Penjualan", "sum"),
    Total_Biaya=("Biaya_Operasional", "sum"),
    Total_Keuntungan=("Keuntungan", "sum"),
    Rata_Rata_Diskon=("Diskon", "mean")
)

print("\nAnalisis berdasarkan jenis pelanggan:")
print(analisis_pelanggan.round(2).to_string())

analisis_pelanggan.to_csv(
    OUTPUT_DIR / "analisis_jenis_pelanggan.csv"
)

ax = analisis_pelanggan["Total_Keuntungan"].plot(
    kind="bar",
    figsize=(8, 4)
)

ax.set_title("Total Keuntungan Berdasarkan Jenis Pelanggan")
ax.set_xlabel("Jenis Pelanggan")
ax.set_ylabel("Keuntungan (Rp)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "keuntungan_pelanggan.png",
    dpi=150
)
plt.show()
plt.close()


# PERSIAPAN DATA MACHINE LEARNING
print("\n" + "=" * 60)
print("3. PREDIKSI KETERLAMBATAN PEMBAYARAN")
print("=" * 60)

fitur = [
    "Jenis_Pelanggan",
    "Penjualan",
    "Diskon",
    "Riwayat_Terlambat",
    "Lama_Berlangganan"
]

X = data[fitur]
y = data["Terlambat_Bayar"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing :", len(X_test))

fitur_kategori = ["Jenis_Pelanggan"]
fitur_numerik = [
    "Penjualan",
    "Diskon",
    "Riwayat_Terlambat",
    "Lama_Berlangganan"
]


preprocessing = ColumnTransformer([
    (
        "kategori",
        OneHotEncoder(handle_unknown="ignore"),
        fitur_kategori
    ),
    (
        "numerik",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        fitur_numerik
    )
])

model = Pipeline([
    ("preprocessing", preprocessing),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
    )
])


# TRAINING DAN PREDIKSI
print("\nMelatih model Random Forest...")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Training dan prediksi selesai.")


# EVALUASI MODEL
print("\n" + "=" * 60)
print("4. EVALUASI MODEL")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

laporan = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Tidak Terlambat", "Terlambat"],
    zero_division=0
)

print("\nClassification Report:")
print(laporan)

# Menyimpan metrik evaluasi
with open(
    OUTPUT_DIR / "evaluasi_model.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("EVALUASI MODEL RANDOM FOREST\n\n")
    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1-score : {f1:.4f}\n\n")
    file.write(laporan)


cm = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
)

tampilan = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Tidak Terlambat", "Terlambat"]
)

tampilan.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix Prediksi Keterlambatan")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=150
)
plt.show()
plt.close()


# IDENTIFIKASI PELANGGAN BERISIKO
print("\n" + "=" * 60)
print("5. IDENTIFIKASI PELANGGAN BERISIKO")
print("=" * 60)

hasil = X_test.copy()
hasil["Pelanggan"] = data.loc[X_test.index, "Pelanggan"]
hasil["Aktual"] = y_test
hasil["Prediksi"] = y_pred

pelanggan_berisiko = hasil[
    hasil["Prediksi"] == 1
].copy()

pelanggan_berisiko["Rekomendasi"] = (
    "Prioritaskan pengingat pembayaran "
    "dan verifikasi tagihan"
)

pelanggan_berisiko = pelanggan_berisiko.sort_values(
    "Riwayat_Terlambat",
    ascending=False
)

pelanggan_berisiko.to_csv(
    OUTPUT_DIR / "pelanggan_berisiko.csv",
    index=False
)

hasil.to_csv(
    OUTPUT_DIR / "hasil_prediksi_testing.csv",
    index=False
)

if pelanggan_berisiko.empty:
    print("\nTidak ada pelanggan yang diprediksi berisiko.")
else:
    print("\nContoh pelanggan yang diprediksi berisiko:")
    print(pelanggan_berisiko.head(15).to_string(index=False))

print("\nJumlah pelanggan berisiko yang terdeteksi:",
      len(pelanggan_berisiko))



# KESIMPULAN
print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)

print(
    "Analisis ini menghasilkan ringkasan penjualan, biaya, "
    "dan keuntungan serta model prediksi keterlambatan pembayaran."
)

print(
    "Periksa grafik dan tabel pada folder outputs untuk "
    "mengetahui hasil analisis secara lebih terperinci."
)

print(
    "\nCatatan: Data yang digunakan merupakan data simulasi, "
    "bukan data transaksi nyata PT Maju Bersama."
)

print("\nProgram selesai dijalankan.")
print("Semua hasil tersimpan di folder:", OUTPUT_DIR.resolve())
