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
from langchain.llms import OpenAI          # ⬅️ OpenRouter kullanımı


# 📁 Dosya yolları
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "datas", "intent_dataset.xlsx")
PDF_PATH   = os.path.join(BASE_DIR, "..", "datas", "depo_sorulari.pdf")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.joblib")
ENC_PATH   = os.path.join(MODEL_DIR, "encoder.joblib")
FAISS_PATH = os.path.join(BASE_DIR, "..", "vectorstore", "faiss_index")

# 🔐 OpenRouter anahtarı
load_dotenv()
OR_KEY = os.getenv("OPENROUTER_KEY")       # .env içindeki key

# 🔁 Ortak embedding modeli
print("🔄 Embed modeli yükleniyor...")
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 🔹 FAISS Index
def create_faiss_index():
    print("📄 PDF yükleniyor...")
    loader = PyMuPDFLoader(PDF_PATH)
    docs   = loader.load()
    chunks = RecursiveCharacterTextSplitter(500, 50).split_documents(docs)
    print("🔍 FAISS index oluşturuluyor...")
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db     = FAISS.from_documents(chunks, embeds)
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    db.save_local(FAISS_PATH)
    print("✅ FAISS index kaydedildi.")

# 🔹 Intent Model
def train_intent_classifier():
    df = pd.read_excel(DATA_PATH).dropna()
    X_vec = embed_model.encode(df["örnek_cümle"].tolist())
    le    = LabelEncoder()
    y_enc = le.fit_transform(df["intent"])
    X_tr, X_te, y_tr, y_te = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    print("🎯 Doğruluk:", accuracy_score(y_te, clf.predict(X_te)))
    print(classification_report(y_te, clf.predict(X_te), target_names=le.classes_))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENC_PATH)
    print("✅ Model & encoder kaydedildi.")
    return clf, le

# 🔹 OpenRouter QA Zinciri
def create_openrouter_qa_chain():
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vect   = FAISS.load_local(FAISS_PATH, embeds, allow_dangerous_deserialization=True)
    llm    = OpenAI(
        openai_api_key = OR_KEY,
        base_url       = "https://openrouter.ai/api/v1",
        model          = "mistralai/mistral-7b-instruct:free",
    )
    prompt = PromptTemplate(
        template = """Sen lojistik destek uzmanısın.
Soru: "{question}"
Belge: {context}
Kısa, net ve profesyonel yanıt ver:""",
        input_variables = ["question", "context"],
    )
    return RetrievalQA.from_chain_type(llm=llm,
                                       retriever=vect.as_retriever(),
                                       chain_type_kwargs={"prompt": prompt},
                                       return_source_documents=False)

# 🔹 Chatbot Döngüsü
def chatbot_loop(clf, le, qa_chain):
    print("\n💬 Depo Asistanı Chatbot (çıkmak için 'çık')\n")
    while True:
        user = input("Siz: ")
        if user.lower() in {"çık", "exit", "q"}:
            print("👋 Görüşmek üzere!")
            break
        intent = le.inverse_transform(clf.predict(embed_model.encode([user])))[0]
        if intent == "greeting":
            print("Bot: Merhaba! Size nasıl yardımcı olabilirim?")
        elif intent == "goodbye":
            print("Bot: Görüşmek üzere!")
        elif intent == "konu_dışı":
            print("Bot: Bu konuda yardımcı olamıyorum. Lütfen lojistikle ilgili bir soru sorun.")
        else:
            print("🔍 Bilgi aranıyor...")
            print("Bot:", qa_chain.run(user))

# 🔹 Ana Akış
if __name__ == "__main__":
    create_faiss_index()
    clf, le = train_intent_classifier()
    qa_chain = create_openrouter_qa_chain()
    chatbot_loop(clf, le, qa_chain)
