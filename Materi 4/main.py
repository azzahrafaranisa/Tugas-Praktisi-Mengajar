
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset simulasi 30 mahasiswa
data = {
    "nilai_rata_rata": [
        85, 78, 90, 65, 72, 88, 60, 75, 92, 68,
        82, 55, 77, 95, 63, 80, 73, 87, 58, 91,
        84, 69, 76, 89, 62, 79, 93, 71, 86, 66
    ],
    "kehadiran": [
        95, 90, 98, 60, 75, 92, 55, 80, 99, 65,
        88, 50, 85, 97, 58, 90, 78, 94, 52, 96,
        91, 70, 82, 93, 60, 87, 98, 73, 92, 68
    ],
    "mata_kuliah_lulus": [
        8, 7, 9, 4, 5, 8, 3, 6, 9, 4,
        7, 2, 6, 9, 3, 7, 5, 8, 2, 9,
        8, 4, 6, 8, 3, 7, 9, 5, 8, 4
    ],
    "lulus_tepat_waktu": [
        1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
        1, 0, 1, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 1, 0, 1, 1, 0, 1, 0
    ]
}

df = pd.DataFrame(data)
print("DATA MAHASISWA")
print(df.to_string(index=False))

# Memisahkan fitur dan target
X = df[["nilai_rata_rata", "kehadiran", "mata_kuliah_lulus"]]
y = df["lulus_tepat_waktu"]

# Membagi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

# Melatih model Decision Tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

print(f"\nAkurasi model: {akurasi:.2%}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["Tidak tepat waktu", "Tepat waktu"],
    zero_division=0
))

# Prediksi mahasiswa baru
mahasiswa_baru = pd.DataFrame([{
    "nilai_rata_rata": 82,
    "kehadiran": 90,
    "mata_kuliah_lulus": 7
}])

prediksi = model.predict(mahasiswa_baru)[0]

print("\nPREDIKSI MAHASISWA BARU")
if prediksi == 1:
    print("Berpotensi lulus tepat waktu")
else:
    print("Berpotensi tidak lulus tepat waktu")

print("\nCatatan: Prediksi menggunakan data simulasi untuk pembelajaran.")
