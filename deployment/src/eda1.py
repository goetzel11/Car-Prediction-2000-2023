import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import pickle 

def run():
    # Membuat title
    st.title('Prediksi Mobil 123')
    st.image('image/mobil.jpg')

    # Sub Header
    st.subheader('1. Backgroup')
    ## Problem Statement
    st.subheader('1.1 Problem Statement')
    st.markdown('Industri otomotif adalah salah satu pasar terbesar yang selalu berubah dan memiliki dinamika harga yang kompleks. Harga mobil bekas tidak ditentukan oleh satu faktor saja, tetapi dipengaruhi oleh berbagai aspek seperti merek, tahun produksi, kondisi kendaraan, ukuran mesin, jenis bahan bakar, jarak tempuh, hingga tipe transmisi. Bagi konsumen maupun dealer mobil, menentukan harga mobil bekas yang akurat adalah tantangan besar. Kesalahan dalam menentukan harga dapat menimbulkan kerugian finansial, baik untuk penjual maupun pembeli. Oleh karena itu, analisis data berbasis machine learning dibutuhkan untuk memprediksi harga secara lebih tepat dan objektif.')

    ## Objective
    st.subheader('1.2 Objective')
    st.markdown('Membantu pengguna dan dealer dalam menentukan estimasi harga mobil tahun 2000 hingga 2023 yang optimal untuk kebutuhan jual/beli agar keputusan lebih cepat, akurat, dan mengurangi risiko kerugian.')

    # EDA
    st.header('2. Exploratory Data Analysis')

    ## 2.1. Brand apa yang sering muncul & bagaimana distribusi harganya?
    st.subheader('2.1. Brand apa yang sering muncul & bagaimana distribusi harganya?')
    st.image('image/Brand_sering.png')
    st.markdown('''
- Dari visualisasi Jumlah Listing per Brand bisa dilihat, bahwa Brand yan sering muncul adalah Toyota dengan jumlah 374, diikuti Audi dengan jumlah 368, dan yang terendah yaitu Ford dengan jumlah 347.
- Berdasarkan boxplot, bentuk boxplot umumnya Tesla memiliki nilai median harga tertinggi. Tesla cenderung brand paling mahal di dataset.
- Toyota, Honda, Ford cenderung lebih murah, Karena median-nya lebih rendah dibanding Audi, BMW, Tesla, Mercedes.
- dan BMW dan Mercedes, memiliki variance yang tinggi atau variasi harga relatif banyak dari yang murah ke mahal.
- Distribusi semua brandnya mirip lebar whisker-nya, harga naik-turunnya tidak terlalu ekstrem per brand, tapi tetap terlihat segmennya.
Boxplot menunjukkan gap yang jelas antara brand premium dan brand mass-market. Tesla memiliki median tertinggi, sedangkan Toyota dan Ford memiliki harga median yang lebih rendah
''')
    
    ## 2.2 Hubungan Tahun produksi vs Harga
    st.subheader('2.2. Brand apa yang sering muncul & bagaimana distribusi harganya?')
    st.image('image/Tahun_vs_harga.png')
    st.markdown('''
- Dapat dilihat pada red line yan menunjukan bahwa Trendline turun, mobil makin tua harga makin murah (depresiasi normal).
- Penyebaran harga sangat besar, jenis mobil sangat bervariasi dalam dataset.
- Tahun pembuatan bukan prediktor kuat harga jika merek/tipe tidak dipisah.
''')

    ## 2.3. Apakah Engine size mempengaruhi harga?
    st.subheader('2.3. Apakah Engine size mempengaruhi harga?')
    st.image('image/Engine_vs_harga.png')
    st.markdown('''
1. Mesin lebih besar sedikit lebih mahal, tetapi efeknya sangat lemah dalam dataset.
2. Harga sangat bervariasi di semua ukuran mesin, faktor lain dominan.
3. Trendline naik tipis, artinya ada hubungan positif, tapi tidak berarti.
''')
    
    ## 2.4. Distribusi Mileage & Hubungan Mileage vs harga
    st.subheader('2.4. Distribusi Mileage & Hubungan Mileage vs harga')
    st.image('image/Distribusi_mileage.png')
    st.markdown('''Analisis visualisasi menunjukkan bahwa meskipun distribusi jarak tempuh (Mileage) kendaraan (rentang 0 hingga 300.000) relatif merata, **harga** (Price) mobil tidak menunjukkan korelasi sama sekali dengan jarak tempuh, garis tren regresi hampir horizontal, dan variasi harga sangat tinggi di semua tingkat jarak tempuh. Anomali ini mengindikasikan bahwa Mileage, yang seharusnya menjadi faktor penentu harga utama di pasar mobil bekas, diabaikan, atau data saat ini kehilangan variabel-variabel kunci yang benar-benar mendorong harga (seperti Model, Tahun Pembuatan, dan Kondisi). Situasi ini menimbulkan risiko kesalahan penetapan harga, di mana mobil berkualitas baik mungkin dijual terlalu murah dan mobil yang sudah tua atau berjarak tempuh tinggi mungkin dihargai terlalu mahal, yang pada akhirnya dapat merugikan profitabilitas bisnis.
''')

    ## 2.5. Perbandingan harga berdasarkan Fuel Type
    st.subheader('2.5. Perbandingan harga berdasarkan Fuel Type')
    st.image('image/Price_by_vuel.png')
    st.markdown('''
Visualisasi Box Plot menunjukkan bahwa, secara keseluruhan, distribusi harga untuk empat jenis bahan bakar (Petrol, Electric, Diesel, Hybrid) memiliki nilai median yang sangat berdekatan (sekitar $50,000 hingga $57,000) dan variabilitas yang luas (dari di bawah $10,000 hingga hampir $100,000). Meskipun demikian, jenis Diesel dan Hybrid menunjukkan sedikit keunggulan, di mana median dan kuartil atas (Q3) mereka berada pada tingkat harga yang sedikit lebih tinggi (mendekati $78,000), menunjukkan bahwa model-model paling mahal dalam inventaris cenderung menggunakan bahan bakar ini. Ketiadaan perbedaan harga yang signifikan pada tingkat median (terutama antara Electric dan Petrol) mengindikasikan bahwa jenis bahan bakar bukanlah faktor penentu harga tunggal, oleh karena itu, strategi harga tidak boleh didasarkan hanya pada jenis bahan bakar, melainkan perlu dipertimbangkan bersama dengan variabel penting lainnya.
Rekomendasi Bisnis Utama:
Validasi Premium Harga: Selidiki mengapa mobil Electric tidak menunjukkan median harga yang lebih tinggi dibandingkan Petrol; ini mungkin menunjukkan peluang untuk menyesuaikan harga jual jika inventaris mobil listrik didominasi oleh model baru atau premium.
Fokus Inventaris Premium: Untuk memaksimalkan pendapatan dari segmen harga atas, prioritaskan akuisisi dan pemasaran model Diesel dan Hybrid karena jenis ini mendominasi kuartil harga tertinggi.
Integrasi Variabel: Gunakan Jenis Bahan Bakar sebagai variabel pendukung (sekunder), dan selalu dasarkan keputusan penetapan harga utama pada Model/Merek dan Tahun Pembuatan, karena ini adalah faktor yang menyebabkan rentang harga yang sangat luas di semua kategori.
                ''')
    ## 2.6. Kondisi Mobil : distribusi & rata-rata harganya
    st.subheader('2.6. Kondisi Mobil : distribusi & rata-rata harganya')
    st.image('image/Distribus_condition.png')
    st.markdown('''Analisis dua visualisasi menunjukkan bahwa inventaris kendaraan memiliki distribusi yang sangat seimbang di antara tiga kategori kondisi utama (New, Used, dan Like New), di mana setiap kategori memiliki jumlah count yang hampir sama (sekitar 810 hingga 850 unit). Keseimbangan inventaris ini menunjukkan basis pasar yang luas, tetapi analisis harga menunjukkan bahwa variabel Jenis Bahan Bakar tidak menyebabkan pemisahan harga yang signifikan; semua jenis bahan bakar (Petrol, Electric, Diesel, Hybrid) memiliki median harga yang berdekatan (antara $50,000 dan $57,000) dengan rentang harga keseluruhan yang sangat lebar (dari minimum hingga $100,000). Meskipun Diesel dan Hybrid menunjukkan sedikit kecenderungan harga yang lebih tinggi di kuartil atas, fakta bahwa harga tidak didorong secara kuat oleh Jenis Bahan Bakar atau Mileage (seperti pada analisis sebelumnya) memperkuat perlunya menggabungkan variabel Kondisi ini bersama dengan Model dan Tahun untuk menciptakan model penetapan harga yang akurat, daripada mengandalkan segmentasi tunggal yang saat ini kurang memisahkan nilai secara efektif.
''')

    ## 2.7. korelasi fitur utama terhadap harga (Heatmap)
    st.subheader('2.7. korelasi fitur utama terhadap harga (Heatmap)')
    st.image('image/Heatmap.png')
    st.markdown('''
Analisis keseluruhan dataset kendaraan bekas menunjukkan adanya anomali signifikan dalam model penetapan harga saat ini, di mana Heatmap Korelasi mengungkapkan bahwa tidak ada variabel kuantitatif (termasuk Year, Engine Size, dan Mileage) yang memiliki korelasi linear yang kuat dengan Price, dengan semua koefisien korelasi sangat mendekati nol. Hal ini menguatkan temuan dari Box Plot Harga vs Jenis Bahan Bakar, yang menunjukkan bahwa semua kategori bahan bakar memiliki median harga yang serupa, membuktikan bahwa variabel prediktif utama untuk harga (seperti Model/Merek) hilang dari analisis kuantitatif ini. Meskipun demikian, data menunjukkan adanya keseimbangan inventaris yang baik antara kategori kondisi (New, Used, dan Like New), yang masing-masing memiliki jumlah unit yang hampir sama, mengindikasikan bahwa bisnis ini memiliki cakupan pasar yang luas namun memerlukan pembaruan segera pada logika penetapan harga dengan memperkaya data dan mengembangkan model multivariat yang mengintegrasikan variabel kualitatif yang hilang.
''')

if __name__ == '__main__':
    run()