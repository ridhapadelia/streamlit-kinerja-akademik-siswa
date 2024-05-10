import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
import pickle
from knn import KNeighborsClassifier

# Load datasets
dataset2 = pd.read_csv("Data Cleanedbisa.csv")

# Sidebar
st.sidebar.title("Dashboard Options")
option = st.sidebar.selectbox("Select Option", ("Business Understanding", "Visualisasi Data", "Prediksi"))

# Main content
st.title("Analisis Kinerja Akademik Siswa Berdasarkan Faktor-Faktor Yang Mempengaruhi Menggunakan Model Klasifikasi")

if option == "Business Understanding":
    st.header("Business Objective")
    st.write("Tujuan bisnis dari analisis ini adalah untuk meningkatkan efektivitas pendidikan dengan memanfaatkan analisis data dan pengembangan model klasifikasi. Dengan memahami faktor-faktor yang mempengaruhi kinerja siswa, diharapkan dapat memberikan pemahaman yang lebih baik tentang perilaku belajar siswa dan faktor-faktor latar belakang yang mempengaruhi hasil akademik. Sehingga institusi pendidikan dapat mengembangkan strategi yang lebih efektif agar dapat menciptakan lingkungan pendidikan yang lebih inklusif dan mampu meningkatkan hasil akademik siswa secara keseluruhan.")
    st.header("Asses Situation")
    st.write("Situasi bisnis yang mendasari analisis ini adalah upaya untuk meningkatkan kinerja akademik siswa di sekolah dengan menganalisis berbagai faktor yang dapat mempengaruhi kinerja siswa baik dari lingkungan sekolah maupun latar belakang kehidupan siswa seperti peran orang tua siswa dalam mendukung kinerja akademiknya.")
    st.header("Data Mining Goals")
    st.write("Analisis kinerja akademik siswa berdasarkan faktor-faktor yang mempengaruhi menggunakan model klasifikasi dalam dataset kinerja akademik siswa memiliki beberapa goals. Pertama, tujuan tersebut adalah untuk membangun model klasifikasi yang dapat memprediksi prestasi akademik siswa dengan tingkat akurasi yang tinggi. Melalui model ini, kita dapat mengidentifikasi faktor-faktor yang paling berpengaruh terhadap kinerja akademik, seperti kehadiran, waktu belajar, dukungan keluarga, dan lainnya. Selanjutnya, analisis tersebut bertujuan untuk membuat segmentasi siswa berdasarkan profil akademik dan non-akademik mereka, memungkinkan penyesuaian strategi pendidikan yang lebih tepat. Selain itu, tujuan lainnya adalah untuk mengeksplorasi pola perilaku siswa yang berkorelasi dengan kinerja akademik, sehingga dapat ditemukan strategi pembelajaran yang lebih efektif.")
    st.header("Project Plan")
    st.write("•	Menganalisis struktur dataset dan memahami arti dari setiap atribut.")
    st.write("•	Menangani missing values, outliers, data yang tidak konsisten dan lain sebagainya")
    st.write("•	Menerapkan metode statistik deskriptif untuk merangkum karakteristik utama dari data.")
    st.write("•	Memvisualisasi data untuk memahami distribusi dan hubungan antar variabel.")
    st.write("•	Memilih model klasifikasi untuk memprediksi kinerja akademik siswa berdasarkan faktor-faktor yang mempengaruhi.")
    st.write("•	Melatih model pada set pelatihan dan mengevaluasi kinerjanya menggunakan metrik yang sesuai (misalnya, akurasi, presisi)")
    st.write("•	Menyesuaikan hyperparameters model untuk meningkatkan kinerja.")
    st.write("•	Melakukan validasi silang untuk menghindari overfitting dan memverifikasi kestabilan model.")
    st.write("•	Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap kinerja akademik siswa berdasarkan analisis model.")
    st.write("•	Menyimpulkan implikasi praktis dan rekomendasi berdasarkan hasil.")
    
elif option == "Visualisasi Data":
    st.header("Visualisasi Data")
    st.write(dataset2)
    st.sidebar.title('Perbandingan Kolom Class dengan kolom lain')
    selected_column = st.sidebar.selectbox('Select a column:', dataset2.columns)

    # Main content
    st.header('Perbandingan Kolom Class dengan ' + selected_column)
    st.bar_chart(dataset2.groupby(['Class', selected_column]).size().unstack())
    st.write("Hasil Kinerja  Akademik Siswa 0=Rendah, 1=Menengah, 2=Tinggi.")
    # Deskripsi kolom yang dipilih
    if selected_column == 'gender':
        st.write("0=Laki-laki, 1=Perempuan.")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi paling banyak adalah siswa perempuan" )
    elif selected_column == 'StageID':
        st.write("0=Tingkat pendidikan dasar, 1=Tingkat pendidikan menengah, 2=Tingkat pendidikan menengah atas")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi paling banyak terdapat pada tingkat pendidikan menengah")
    elif selected_column == 'GradeID':
        st.write("0=G-02, 1=G-04, 2=G-05, 3=G-06, 4=G-07, 5=G-08, 6=G-09, 7=G-10, 8=G-11, 9=G-12")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi paling banyak terdapat pada id kelas G-02 dan G-08")
    elif selected_column == 'SectionID':
        st.write("0=A, 1=B, 2=C")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi paling banyak terdapat pada kelas A")
    elif selected_column == 'Topic':
        st.write("0=IT, 1=Math, 2=Arabic, 3=Science, 4=English, 5=Quran, 6=Spanish, 7=French, 8=History, 9=Biology, 10=Chemistry, 11=Geology")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi paling banyak di mata pelajaran Arabic dan French") 
    elif selected_column == 'Semester':
        st.write("0=Semester pertama, 1=Semester kedua")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi paling banyak di semester 2")
    elif selected_column == 'Relation':
        st.write("0=Ayah, 1=Ibu")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi, orang tua yang bertanggung jawab adalah Ibu ")
    elif selected_column == 'raisehands':
        st.write("kolom raisehands ini berisi informasi mengenai berapa kali siswa tersebut mengangkat tangannya di depan kelas untuk menjawab atau menanyakan pertanyaan dengan rentang 0 hingga 100")
    elif selected_column == 'VisITedResources':
        st.write("kolom VisITedResources ini berisi informasi mengenai berapa kali siswa mengunjungi konten kursus dengan rentang 0 hingga 100")
    elif selected_column == 'Discussion':
        st.write("kolom Discussion ini berisi informasi mengenai berapa kali siswa berpartisipasi dalam kelompok diskusi dengan rentang 0 hingga 100")
    elif selected_column == 'StudentAbsenceDays':
        st.write("0=Di bawah 7 hari, 1=di atas 7 hari")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi paling banyak adalah siswa dengan jumlah absensi ketidakhadirannya di bawah 7 hari")
    elif selected_column == 'GradeCategory':
        st.write("0=Anak-anak, 1=Remaja")
        st.write("Dari visualisasi di atas, dapat terlihat bahwa siswa dengan hasil kinerja akademik tinggi paling banyak adalah siswa dengan kategori umur remaja")


    st.header("Visualisasi Data")
    ## DISTRIBUSI Tingkat Pendidikan Siswa
    plt.figure(figsize=(8, 6))
    sns.histplot(dataset2['StageID'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribusi Tingkat Pendidikan Siswa')
    plt.xlabel('Tingkat Pendidikan')
    plt.ylabel('Frekuensi')
    plt.grid(True)
    # Menampilkan plot di Streamlit
    st.pyplot(plt)
    st.write("Visualisasi di atas ini menunjukkan tingkat pendidikan siswa. 0=lowerlevel(pendidikan dasar), 1=MiddleSchool(pendidikan tingkat menengah), 2=HighLevel(pendidikan tingkat menengah atas). Dapat terlihat bahwa jenjang pendidikan menengah memiliki frekuensi paling besar ini artinya kebanyakan siswa berasal dari tingkat pendidikan menengah, kemudian tingkat pendidikan dasar, dan frekuensi paling sedikit di tingkat pendidikan menengah atas.")

    # Visualisasi countplot dengan seaborn
    plt.figure(figsize=(8, 6))
    sns.countplot(x='StudentAbsenceDays', data=dataset2, palette='Set2')
    plt.xlabel('StudentAbsenceDays')
    plt.ylabel('Frekuensi')
    plt.title('Jumlah Hari Absensi Siswa')
    st.pyplot(plt)
    st.write("Visualisasi di atas ini menunjukkan banyaknya hari siswa tidak hadir. 0=Under-7(di bawah 7 hari) dan 1=Above-7(di atas 7 hari). Dapat terlihat bahwa rata-rata siswa absen di bawah 7 hari yaitu dengan frekunsi sekitar 280 hingga 290 siswa. dan sisanya sekitar 180 hingga 190 siswa absen di atas 7 hari.")

    ## DIAGRAM PIE GENDER
    gender_counts = dataset2['gender'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#1f77b4', '#ff7f0e'])
    plt.title('Gender Siswa')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(plt)
    st.write("Visualisasi di atas ini menunjukkan persentase jenis kelamin siswa. 0=M(laki-laki) 1=F(perempuan). Dapat terlihat bahwa mayoritas siswa berjenis kelamin laki-laki dengan persentase sebesar 63.4% dan sisanya sebesar 36.6% merupakan siswa perempuan.")

elif option == "Prediksi":
    file_path = 'knn.pkl'

    with open(file_path , 'rb') as f:
        clf = joblib.load(f)
    st.title("Prediksi Kinerja Akademik Siswa")
    def preprocess_data(data):
        # Preprocessing: Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        return data_scaled
    
    gender = st.selectbox('gender', ['M', 'F'])
    SectionID = st.selectbox('SectionID', ['A', 'B', 'C'])
    VisITedResources = int(st.number_input('VisitedResources', 0,100, 0))
    Relation = st.selectbox('Relation', ['Father', 'Mum'])
    StageID = st.selectbox('StageID', ['lowerlevel', 'MiddleSchool', 'HighSchool'])
    Semester = st.selectbox('Semester', ['F', 'S'])
    Discussion = int(st.number_input('Discussion', 0,100, 0))
    Topic = st.selectbox('Topic', ['IT', 'Math', 'Arabic', 'Science', 'English', 'Quran', 'Spanish', 'French', 'History', 'Biology', 'Chemistry', 'Geology' ])
    GradeID = int(st.selectbox('GradeID', [0,1,2,3,4,5,6,7,8,9]))
    GradeCategory = st.selectbox('GradeCategory',['Child','Teenager'])
    StudentAbsenceDays = st.selectbox('StudentAbsenceDays',['Under-7','Above-7'])
    raisedhands = int(st.number_input('raisedhands', 0,100, 0))
    
    prediction_state = st.markdown('calculating')

    Class = pd.DataFrame({ 
        'gender' : [gender],
        'SectionID' : [SectionID],
        'VisITedResources' : [VisITedResources],
        'Relation' : [Relation],
        'StageID' : [StageID],
        'Semester' : [Semester],
        'Discussion' : [Discussion],
        'Topic' : [Topic],
        'GradeID' : [GradeID],
        'GradeCategory' : [GradeCategory],
        'StudentAbsenceDays' : [StudentAbsenceDays],
        'raisedhands' : [raisedhands]
    })



    x_pred = clf.predict(Class)

    if x_pred[0] == 0:
        msg = 'Low-Level'
    elif x_pred[0] == 1:
        msg = 'Middle-Level'
    elif x_pred[0] == 2:
        msg = 'High-Level'
    else:
        msg = 'Unknown'

    prediction_state.markdown(msg)

