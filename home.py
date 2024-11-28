import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Judul aplikasi
st.title("Prediksi Klik Iklan")

# Gunakan dataset internal atau dataset yang sudah ada
# Contoh dataset kecil sebagai data internal
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'gender': ['Pria', 'Wanita', 'Pria', 'Wanita', 'Pria'],
    'device_type': ['Desktop', 'Mobile', 'Desktop', 'Mobile', 'Desktop'],
    'ad_position': ['Atas', 'Bawah', 'Tengah', 'Atas', 'Bawah'],
    'time_of_day': ['Pagi', 'Siang', 'Malam', 'Pagi', 'Siang'],
    'click': [0, 1, 0, 1, 0]
})

# 2. Exploratory Data Analysis (EDA)
st.subheader("Informasi Dataset")
st.write(data.info())

st.subheader("Statistik Deskriptif")
st.write(data.describe())

st.subheader("Distribusi Nilai Kosong per Kolom")
st.write(data.isnull().sum())

# 3. Visualisasi Data
st.subheader("Distribusi Usia")
plt.figure(figsize=(8, 5))
sns.histplot(data['age'], kde=True, bins=20, color='blue')
plt.title("Distribusi Usia")
plt.xlabel("Usia")
plt.ylabel("Frekuensi")
st.pyplot()

st.subheader("Distribusi Gender")
plt.figure(figsize=(6, 4))
data['gender'].value_counts().plot(kind='bar', color=['pink', 'lightblue'])
plt.title("Distribusi Gender")
plt.xlabel("Gender")
plt.ylabel("Jumlah")
st.pyplot()

# 4. Click-through rate (CTR) berdasarkan perangkat
st.subheader("Click-through Rate Berdasarkan Perangkat")
plt.figure(figsize=(8, 5))
sns.barplot(x='device_type', y='click', data=data, estimator=np.mean, errorbar=None)
plt.title("Click-through Rate Berdasarkan Perangkat")
plt.xlabel("Tipe Perangkat")
plt.ylabel("CTR (Click Rate)")
st.pyplot()

# 5. Preprocessing Data
encoder = LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])
data['device_type'] = encoder.fit_transform(data['device_type'])
data['ad_position'] = encoder.fit_transform(data['ad_position'])
data['time_of_day'] = encoder.fit_transform(data['time_of_day'])

# 6. Membagi Dataset
X = data.drop('click', axis=1)
y = data['click']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menghapus baris yang mengandung NaN
data = data.dropna()
X = data.drop('click', axis=1)
y = data['click']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputer untuk kolom numerik
imputer_num = SimpleImputer(strategy='mean')
X_train = imputer_num.fit_transform(X_train)
X_test = imputer_num.transform(X_test)

# Imputer untuk kolom kategorikal (jika ada kolom kategorikal)
imputer_cat = SimpleImputer(strategy='most_frequent')
X_train = imputer_cat.fit_transform(X_train)
X_test = imputer_cat.transform(X_test)

# 7. Membuat dan Melatih Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 8. Evaluasi Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.subheader(f"Akurasi Model: {accuracy * 100:.2f}%")

st.subheader("Classification Report")
class_report = classification_report(y_test, y_pred, output_dict=True)
st.write(class_report)

st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# Visualisasi Confusion Matrix
st.subheader("Visualisasi Confusion Matrix")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Klik', 'Klik'], yticklabels=['Tidak Klik', 'Klik'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot()

# 9. Menyimpan Model
filename = 'ad_prediction_model.sav'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
st.write(f"Model berhasil disimpan ke file {filename}")

# 10. Memuat Model dan Prediksi Baru
st.subheader("Prediksi Baru")
# Form input untuk prediksi
age = st.number_input("Usia", min_value=18, max_value=100, value=25)
gender = st.selectbox("Gender", options=["Pria", "Wanita"])
device_type = st.selectbox("Tipe Perangkat", options=["Desktop", "Mobile"])
ad_position = st.selectbox("Posisi Iklan", options=["Atas", "Bawah", "Tengah"])
time_of_day = st.selectbox("Waktu Iklan", options=["Pagi", "Siang", "Malam"])

# Mengonversi input menjadi format yang sesuai
input_data = np.array([[age, gender, device_type, ad_position, time_of_day]])

# Pastikan LabelEncoder dapat menangani input baru
def safe_transform(encoder, column_data):
    # Jika label belum terlihat sebelumnya, kita akan memetakan nilai yang tidak dikenal
    try:
        return encoder.transform(column_data)
    except ValueError:
        # Jika ada label yang belum terlihat, kita akan menambahkan label tersebut
        return encoder.fit_transform(column_data)

# Menerapkan safe_transform pada setiap kolom kategorikal
input_data[0, 1] = safe_transform(encoder, input_data[0, 1:2])[0]  # gender
input_data[0, 2] = safe_transform(encoder, input_data[0, 2:3])[0]  # device_type
input_data[0, 3] = safe_transform(encoder, input_data[0, 3:4])[0]  # ad_position
input_data[0, 4] = safe_transform(encoder, input_data[0, 4:5])[0]  # time_of_day

if st.button('Prediksi Klik'):
    # Memuat model yang disimpan
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    
    # Prediksi dengan model yang telah dimuat
    prediction = loaded_model.predict(input_data)
    
    if prediction == 1:
        st.success("Iklan akan diklik!")
    else:
        st.warning("Iklan tidak akan diklik.")
