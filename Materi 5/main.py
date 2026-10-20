
import pandas as pd
import matplotlib.pyplot as plt

# Data penjualan
data = {
    "Bulan": [
        "Januari", "Januari", "Februari", "Februari",
        "Maret", "Maret", "April", "April",
        "Mei", "Mei", "Juni", "Juni"
    ],
    "Produk": [
        "Laptop", "Aksesori", "Laptop", "Aksesori",
        "Laptop", "Aksesori", "Laptop", "Aksesori",
        "Laptop", "Aksesori", "Laptop", "Aksesori"
    ],
    "Jumlah_Terjual": [15, 30, 18, 35, 20, 32, 25, 40, 22, 45, 28, 50],
    "Harga": [7000000, 150000] * 6
}

df = pd.DataFrame(data)
df["Pendapatan"] = df["Jumlah_Terjual"] * df["Harga"]

print("DATA PENJUALAN")
print(df.to_string(index=False))

print(f"\nTotal pendapatan: Rp{df['Pendapatan'].sum():,.0f}")
print(f"Total produk terjual: {df['Jumlah_Terjual'].sum()} unit")

terlaris = df.groupby("Produk")["Jumlah_Terjual"].sum().idxmax()
print("Produk terlaris:", terlaris)

# Grafik total penjualan per produk
df.groupby("Produk")["Jumlah_Terjual"].sum().plot(
    kind="bar", title="Total Penjualan Berdasarkan Produk"
)
plt.xlabel("Produk")
plt.ylabel("Jumlah Terjual")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Filter produk melalui terminal
print("\nPilihan produk: Semua Produk, Laptop, Aksesori")
pilihan = input("Masukkan pilihan produk: ").strip().title()

if pilihan.lower() == "semua produk":
    hasil = df.copy()
elif pilihan in df["Produk"].unique():
    hasil = df[df["Produk"] == pilihan].copy()
else:
    print("Pilihan tidak valid. Menampilkan semua produk.")
    hasil = df.copy()
    pilihan = "Semua Produk"

print(f"\nDASHBOARD: {pilihan}")
print(f"Total pendapatan: Rp{hasil['Pendapatan'].sum():,.0f}")
print(f"Total produk terjual: {hasil['Jumlah_Terjual'].sum()} unit")

urutan_bulan = [
    "Januari", "Februari", "Maret",
    "April", "Mei", "Juni"
]

pendapatan = (
    hasil.groupby("Bulan")["Pendapatan"]
    .sum()
    .reindex(urutan_bulan, fill_value=0)
)

pendapatan.plot(kind="bar", figsize=(9, 5))
plt.title(f"Pendapatan per Bulan - {pilihan}")
plt.xlabel("Bulan")
plt.ylabel("Pendapatan (Rp)")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
