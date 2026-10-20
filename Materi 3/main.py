
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Data simulasi wilayah
wilayah = pd.DataFrame({
    "nama_wilayah": ["Wilayah A", "Wilayah B", "Wilayah C",
                     "Wilayah D", "Wilayah E", "Wilayah F"],
    "latitude": [-3.790, -3.800, -3.815, -3.825, -3.780, -3.810],
    "longitude": [102.260, 102.280, 102.250, 102.290, 102.310, 102.320],
    "jumlah_penduduk": [12000, 18000, 9000, 22000, 15000, 25000]
})

# Data fasilitas umum yang tersedia
fasilitas = pd.DataFrame({
    "nama_fasilitas": ["Fasilitas 1", "Fasilitas 2", "Fasilitas 3"],
    "latitude": [-3.795, -3.820, -3.785],
    "longitude": [102.265, 102.300, 102.315]
})

# Menghitung jarak menggunakan rumus Haversine
def hitung_jarak(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)

    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Menentukan fasilitas terdekat setiap wilayah
jarak_terdekat = []

for _, w in wilayah.iterrows():
    jarak = [
        hitung_jarak(w["latitude"], w["longitude"],
                     f["latitude"], f["longitude"])
        for _, f in fasilitas.iterrows()
    ]
    indeks = np.argmin(jarak)
    jarak_terdekat.append({
        "nama_fasilitas": fasilitas.iloc[indeks]["nama_fasilitas"],
        "jarak_km": jarak[indeks]
    })

hasil = pd.concat(
    [wilayah, pd.DataFrame(jarak_terdekat)], axis=1
)

# Menentukan prioritas pembangunan
hasil["status"] = np.where(
    (hasil["jumlah_penduduk"] >= 15000) & (hasil["jarak_km"] > 1.0),
    "Prioritas", "Bukan prioritas"
)

print("HASIL ANALISIS WILAYAH")
print(hasil[["nama_wilayah", "jumlah_penduduk",
             "nama_fasilitas", "jarak_km", "status"]]
      .round({"jarak_km": 2}).to_string(index=False))

# Visualisasi wilayah dan fasilitas
plt.figure(figsize=(10, 7))

for _, w in hasil.iterrows():
    warna = "red" if w["status"] == "Prioritas" else "blue"
    plt.scatter(w["longitude"], w["latitude"], color=warna, s=100)
    plt.annotate(w["nama_wilayah"],
                 (w["longitude"], w["latitude"]),
                 xytext=(5, 5), textcoords="offset points")

for _, f in fasilitas.iterrows():
    plt.scatter(f["longitude"], f["latitude"],
                color="green", marker="^", s=160)
    plt.annotate(f["nama_fasilitas"],
                 (f["longitude"], f["latitude"]),
                 xytext=(5, -12), textcoords="offset points")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Sebaran Wilayah dan Fasilitas Umum")
plt.legend(handles=[
    Line2D([], [], marker="o", color="w",
           markerfacecolor="red", label="Prioritas", markersize=9),
    Line2D([], [], marker="o", color="w",
           markerfacecolor="blue", label="Bukan prioritas", markersize=9),
    Line2D([], [], marker="^", color="w",
           markerfacecolor="green", label="Fasilitas", markersize=9)
])
plt.grid(True)
plt.tight_layout()
plt.show()

prioritas = hasil[hasil["status"] == "Prioritas"]
print("\nWilayah yang diprioritaskan:")
print(prioritas[["nama_wilayah", "jumlah_penduduk", "jarak_km"]]
      .round({"jarak_km": 2}).to_string(index=False))
print("\nJumlah wilayah prioritas:", len(prioritas))

print("\nKesimpulan: Analisis spasial dapat membantu menentukan")
print("prioritas lokasi fasilitas umum berdasarkan data simulasi.")
print("Hasil ini perlu divalidasi dengan data dan kondisi lapangan nyata.")
