
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

np.random.seed(42)

# Membuat titik aman dan posisi masyarakat
titik_aman = np.array([[100, 100], [900, 100], [500, 900]])
masyarakat = np.random.uniform(0, 1000, (100_000, 2))

print("Jumlah masyarakat:", len(masyarakat))
print("Jumlah titik aman:", len(titik_aman))

# Visualisasi wilayah evakuasi
plt.scatter(masyarakat[:3000, 0], masyarakat[:3000, 1],
            s=8, alpha=0.4, label="Masyarakat")
plt.scatter(titik_aman[:, 0], titik_aman[:, 1],
            color="red", marker="*", s=200, label="Titik Aman")
plt.title("Simulasi Wilayah Evakuasi")
plt.xlabel("Koordinat X (meter)")
plt.ylabel("Koordinat Y (meter)")
plt.legend()
plt.grid()
plt.show()

# Metode 1: Python Loop
def hitung_loop(data):
    hasil = []
    for posisi in data:
        jarak = np.sqrt(np.sum((titik_aman - posisi) ** 2, axis=1))
        hasil.append(np.min(jarak))
    return np.array(hasil)

# Metode 2: NumPy
def hitung_numpy(data):
    selisih = data[:, None, :] - titik_aman[None, :, :]
    jarak = np.sqrt(np.sum(selisih ** 2, axis=2))
    return np.min(jarak, axis=1)

# Membandingkan waktu komputasi
data_uji = masyarakat[:10_000]

mulai = time.perf_counter()
hasil_loop = hitung_loop(data_uji)
waktu_loop = time.perf_counter() - mulai

mulai = time.perf_counter()
hasil_numpy = hitung_numpy(data_uji)
waktu_numpy = time.perf_counter() - mulai

print(f"\nWaktu Python Loop: {waktu_loop:.4f} detik")
print(f"Waktu NumPy      : {waktu_numpy:.4f} detik")
print("Hasil sama:", np.allclose(hasil_loop, hasil_numpy))

# Simulasi beberapa skenario
jarak_masyarakat = hitung_numpy(masyarakat)
skenario = [1000, 10000, 50000, 100000]
hasil = []

for jumlah in skenario:
    jarak = jarak_masyarakat[:jumlah]
    waktu = np.max(jarak) / 1.2 * 1.5 / 60

    hasil.append({
        "Jumlah Masyarakat": jumlah,
        "Jarak Rata-rata (m)": round(np.mean(jarak), 2),
        "Jarak Terjauh (m)": round(np.max(jarak), 2),
        "Estimasi Waktu (menit)": round(waktu, 2)
    })

df = pd.DataFrame(hasil)
print("\nHASIL SIMULASI")
print(df.to_string(index=False))

# Grafik estimasi waktu
plt.plot(df["Jumlah Masyarakat"],
         df["Estimasi Waktu (menit)"], marker="o")
plt.title("Estimasi Waktu Evakuasi")
plt.xlabel("Jumlah Masyarakat")
plt.ylabel("Waktu Terlama (menit)")
plt.grid()
plt.show()

print("\nKESIMPULAN")
print("Jumlah titik aman:", len(titik_aman))
print(f"Jarak rata-rata: {np.mean(jarak_masyarakat):.2f} meter")
print(f"Jarak terjauh: {np.max(jarak_masyarakat):.2f} meter")
print("Simulasi ini untuk pembelajaran, bukan evakuasi nyata.")
