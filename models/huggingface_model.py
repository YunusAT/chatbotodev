import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# YEREL LLM İÇİN YENİ İMPORTLAR
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch  # GPU kontrolü için

# 📁 Dosya yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "datas", "intent_dataset.xlsx")
PDF_PATH     = os.path.join(BASE_DIR, "datas", "depo_sorulari.pdf")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "intent_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")
FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

load_dotenv()

# 🔁 Ortak embedding modeli
print("🔄 Embedding modeli yükleniyor...")
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# 🔹 FAISS Index Oluşturma
def create_faiss_index():
    print("📄 PDF yükleniyor...")
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    print("🔍 FAISS index oluşturuluyor...")
    # Embedding modeli burada da aynı olmalı
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_documents(texts, embeddings)
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    db.save_local(FAISS_PATH)
    print(f"✅ FAISS index kaydedildi: {FAISS_PATH}")


# 🔹 Intent Model Eğitimi
def train_intent_classifier():
    df = pd.read_excel(DATASET_PATH).dropna()
    X, y = df["örnek_cümle"].tolist(), df["intent"].tolist()
    X_vec = embed_model.encode(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.2, stratify=y_encoded,
                                                        random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("🎯 Doğruluk:", accuracy_score(y_test, y_pred))
    print("\n📊 Rapor:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("✅ Model ve encoder kaydedildi.")
    return clf, le


# 🔹 RAG Zinciri Oluştur (Yerel bir Hugging Face modeli kullanılacak şekilde değiştirildi)
def create_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Örnek: Mistral 7B Instruct

    # GPU varlığını kontrol et ve cihazı ayarla
    device = 0 if torch.cuda.is_available() else -1  # 0 GPU, -1 CPU
    print(f"LLM, cihaz üzerinde yükleniyor: {'CUDA (GPU)' if device == 0 else 'CPU'}")

    try:
        # Nicemleme (Quantization) ayarları: Bellek kullanımını azaltır.
        # Daha az belleğe sahipseniz 'load_in_4bit=True' veya 'load_in_8bit=True' kullanın.
        # Büyük modellerde bu zorunlu olabilir.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit nicemleme
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Modeli ve tokenleştiriciyi yükle
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config if torch.cuda.is_available() else None,  # GPU varsa nicemleme kullan
            device_map="auto" if torch.cuda.is_available() else None,  # GPU varsa otomatik cihaz eşleme
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            # CPU üzerinde çalıştırırken nicemlemeyi kapatın, çünkü BitsAndBytes sadece GPU içindir.
        )
        model.eval()  # Modeli değerlendirme moduna al

        # Metin üretimi için Hugging Face pipeline oluştur
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            # max_new_tokens: Üretilecek maksimum token sayısı
            max_new_tokens=500,
            # do_sample: Rastgele örnekleme yapılıp yapılmayacağı (True=daha yaratıcı, False=daha deterministik)
            do_sample=True,
            # temperature: Çıktının rastgeleliğini kontrol eder (düşük=daha odaklı, yüksek=daha çeşitli)
            temperature=0.7,
            # top_k: En olası k token arasından seçim yapar
            top_k=50,
            # top_p: Toplam olasılığı p olan token grubundan seçim yapar (çekirdek örnekleme)
            top_p=0.95,
            # num_return_sequences: Üretilecek bağımsız sıra sayısı (genellikle 1)
            num_return_sequences=1,
            # eos_token_id: Cümlenin sonu token ID'si, üretimi durdurmak için
            eos_token_id=tokenizer.eos_token_id
        )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print(f"✅ Yerel LLM '{model_id}' başarıyla yüklendi.")

    except Exception as e:
        print(f"❌ Yerel LLM yüklenirken bir hata oluştu: {e}")
        print("Lütfen yeterli bellek (RAM/VRAM) olduğundan, gerekli kütüphanelerin yüklü olduğundan")
        print("ve model ID'sinin doğru olduğundan emin olun. Daha küçük bir model deneyebilirsiniz.")
        return None  # Hata durumunda None döndür

    # --- YEREL LLM ENTEGRASYONU SONU ---

    prompt = PromptTemplate(
        template="""Sen lojistik destek konusunda uzman bir yardımcısın.
Kullanıcı şu soruyu sordu: "{question}"
Aşağıdaki belgeleri oku ve net, kısa ve profesyonel bir yanıt ver:

{context}
Yanıt:""",  # Yanıtın başlangıcını belirtmek için "Yanıt:" ekledim
        input_variables=["question", "context"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )


# 🔹 Chatbot Döngüsü
def chatbot_loop(clf, le, qa_chain):
    print("\n💬 Lojistik Destek Chatbotu (çıkmak için 'çık', 'exit' veya 'q' yazın)\n")
    if qa_chain is None:
        print("Chatbot başlatılamadı çünkü LLM yüklenirken bir sorun oluştu.")
        return

    while True:
        user_input = input("Siz: ")
        if user_input.lower() in ["çık", "exit", "q"]:
            print("👋 Görüşmek üzere!")
            break

        # Kullanıcı girdisini vektörleştir ve niyeti tahmin et
        vec = embed_model.encode([user_input])
        intent = le.inverse_transform(clf.predict(vec))[0]

        print(f"Tahmin edilen niyet: {intent}")  # Niyetin ne olduğunu görmek için

        if intent == "greeting":
            print("Bot: Merhaba! Size nasıl yardımcı olabilirim?")
        elif intent == "goodbye":
            print("Bot: Görüşmek üzere!")
        elif intent == "konu_dışı":
            print("Bot: Bu konuda yardımcı olamıyorum. Lütfen lojistikle ilgili bir şey sorun.")
        else:
            print("🔍 Bilgi aranıyor...")
            try:
                # QA zincirini çalıştır
                response = qa_chain.run(user_input)
                # Model çıktısında oluşan potansiyel istem tekrarını temizle
                if response.startswith(user_input):
                    response = response[len(user_input):].strip()
                # Bazen model "Yanıt:" gibi istenmeyen metinler üretebilir, onları temizle
                if response.startswith("Yanıt:"):
                    response = response[len("Yanıt:"):].strip()
                print("Bot:", response.strip())  # Fazla boşlukları temizle
            except Exception as e:
                print(f"Bot: Yanıt oluşturulurken bir hata oluştu: {e}")
                print("Lütfen LLM'in doğru yüklendiğinden ve çalıştığından emin olun.")


# 🔹 Ana Akış
if __name__ == "__main__":
    create_faiss_index()  # FAISS indeksini oluştur
    clf, le = train_intent_classifier()  # Niyet sınıflandırıcıyı eğit
    qa_chain = create_qa_chain()  # RAG zincirini oluştur (yerel LLM ile)
    chatbot_loop(clf, le, qa_chain)  # Chatbot döngüsünü başlat