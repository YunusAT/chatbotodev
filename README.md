# 📦 Akıllı Depo Asistanı – Chatbot Projesi

Bu çalışma, lojistik destek kapsamında sıkça karşılaşılan sorulara otomatik ve tutarlı yanıtlar sunmak amacıyla geliştirilmiş bir yapay zekâ tabanlı sohbet robotunu içermektedir. Model, dökümanlardan bilgi çıkarma ve kullanıcı mesajlarını anlama üzerine kurgulanmıştır.

---

## 🔧 Projeye Genel Bakış

- Kullanıcı mesajlarını anlayabilen bir sınıflandırıcı (niyet tanıma)
- PDF formatındaki bilgilendirme dosyalarından bilgi çekme
- FAISS altyapısı ile vektör tabanlı arama
- OpenRouter aracılığıyla Mistral-7B gibi büyük dil modelleriyle entegrasyon
- Web tabanlı arayüz (Streamlit ile)

---

## 🚀 Kurulum ve Çalıştırma Adımları

### 📌 Adım 1: Gerekli Kütüphanelerin Yüklenmesi
Terminale aşağıdaki komutu yaz:

```bash
pip install -r requirements.txt
```

### 🔐 Adım 2: Ortam Değişkeni (API Key)
Proje kök dizinine `.env` adında bir dosya oluştur. İçine şu satırı ekle:

```
OPENROUTER_KEY=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
```

### 🧠 Adım 3: Model Eğitimi ve Vektör Veri Üretimi
Aşağıdaki komut ile eğitim işlemini başlat:

```bash
python train_and_index.py
```

### 💻 Adım 4: Uygulama Arayüzünü Başlat
Chatbot'u başlatmak için şu komutu kullan:

```bash
streamlit run app/app.py
```

---

## 🗂️ Klasör ve Dosya Düzeni

```
chatbot/
├── app/
│   └── app.py
├── datas/
│   ├── intent_dataset.xlsx
│   └── depo_sorulari.pdf
├── models/
│   ├── intent_model.joblib
│   └── encoder.joblib
├── vectorstore/
│   └── faiss_index/
├── .env
└── requirements.txt
```

---

## 📈 Model Değerlendirmesi

### 🎯 Intent Sınıflandırma
%80 eğitim / %20 test verisi kullanılarak değerlendirildi.

| Model               | Precision | Recall | F1 Score |
|---------------------|-----------|--------|----------|
| Logistic Regression | 0.96      | 0.95   | 0.955    |

### 🤖 Dil Modeli Karşılaştırması

| Model                    | Doğruluk | Kalite | BERTScore |
|--------------------------|----------|--------|-----------|
| Mistral-7B (OpenRouter)  | ✅✅✅✅     | ✅✅✅   | 0.765     |
| LLaMA-2                  | ✅✅✅✅     | ✅✅✅✅  | 0.870     |

---

## 💬 Örnek Sohbet

| Kullanıcı Sorusu  | Anlaşılan Niyet | Yanıt                                         |
|-------------------|------------------|-----------------------------------------------|
| merhaba           | greeting         | Merhaba! Size nasıl yardımcı olabilirim?     |
| çalışma saatleri  | bilgi            | Depomuz hafta içi 09:00 - 18:00 arası açıktır. |
| görüşürüz         | goodbye          | Görüşmek üzere, iyi günler!                  |

---

## ❗️ Sık Karşılaşılan Hatalar

| Problem                         | Açıklama                                          |
|----------------------------------|---------------------------------------------------|
| `Model dosyası eksik`           | Eğitim komutu (`train_and_index.py`) çalıştırılmalı |
| `FAISS index bulunamadı`        | Vektör dizini eksikse yeniden oluşturulmalı       |
| `OPENROUTER_KEY eksik`          | `.env` dosyası kontrol edilmeli                   |

---

## 🛠️ Geliştirme İçin Öneriler

- Veritabanı destekli sorgulama
- Chat geçmişi ve kullanıcı oturumu takibi
- Intent modeli olarak transformer kullanımı
- Daha gelişmiş yanıt kalitesi için RAG iyileştirmesi

---

## 👤 Hazırlayan

**Yunus Alp Turan – 171421001**

> Bu proje, depo süreçlerini destekleyen sorulara hızlı ve akıllı cevaplar üretmek amacıyla tasarlanmıştır.
