import streamlit as st
import os
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  # ⬅️ OpenRouter üzerinden LLM
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch

# ──────────────────────────────────────────────
# ⚙️ 1. Sayfa Konfigürasyonu
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Depo Asistanı Chatbot",
    page_icon="📦",
    layout="wide",
)

# ──────────────────────────────────────────────
# 🎨 2. Tema / CSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
    .stApp {background:#ffffff;}
    .chat-message {padding:1rem;border-radius:10px;margin-bottom:1rem;border-left:4px solid #6f42c1;color:#000;}
    .user-message {background:#f3e8ff;border-left-color:#6f42c1;}
    .assistant-message {background:#e8f5e9;border-left-color:#2e7d32;}
    .stButton>button {background:#6f42c1;color:#fff;border-radius:6px;border:none;padding:.5rem 1rem;}
    .stButton>button:hover {background:#533089;}
    .title {color:#4a235a;text-align:center;margin-bottom:2rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 📂 3. Dosya ve Ortam Ayarları
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "datas" / "lojistik_50_sss.pdf"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FAISS_PATH = BASE_DIR / "vectorstore" / "faiss_index"

load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")  # .env içine yazdığın anahtar

# ──────────────────────────────────────────────
# 🗄️ 4. Cache'li Yardımcılar
# ──────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_intent_classifier():
    if MODEL_PATH.exists() and ENCODER_PATH.exists():
        return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)
    return None, None

@st.cache_resource
def ensure_faiss():
    if FAISS_PATH.exists():
        return True
    if not PDF_PATH.exists():
        st.error(f"PDF bulunamadı: {PDF_PATH}")
        return False
    loader = PyMuPDFLoader(str(PDF_PATH))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_documents(chunks, embeds)
    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(FAISS_PATH))
    return True

# ──────────────────────────────────────────────
# 🤖 5. QA Zincirleri
# ──────────────────────────────────────────────

def create_openrouter_qa_chain():
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vect = FAISS.load_local(str(FAISS_PATH), embeds, allow_dangerous_deserialization=True)
    llm = OpenAI(
        openai_api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/mistral-7b-instruct:free",
    )
    prompt = PromptTemplate(
        template="""Sen lojistik destek uzmanısın.\nSoru: \"{question}\"\nBelge: {context}\nKısa ve net yanıt ver:""",
        input_variables=["question", "context"],
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=vect.as_retriever(), chain_type_kwargs={"prompt": prompt})


def create_local_llm_qa_chain():
    embeds = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vect = FAISS.load_local(str(FAISS_PATH), embeds, allow_dangerous_deserialization=True)
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    device = 0 if torch.cuda.is_available() else -1
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if device == 0 else None
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto" if device == 0 else None, quantization_config=bnb_cfg)
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=500, temperature=0.7, top_p=0.95)
    llm = HuggingFacePipeline(pipeline=pipe)
    prompt = PromptTemplate(template="""Sen lojistik destek uzmanısın.\nSoru: \"{question}\"\nBelge: {context}\nYanıt:""", input_variables=["question", "context"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=vect.as_retriever(), chain_type_kwargs={"prompt": prompt})

# ──────────────────────────────────────────────
# 💬 6. Intent Kısa Yanıtları
# ──────────────────────────────────────────────
INTENT_SHORT = {
    "greeting": "Merhaba! Size nasıl yardımcı olabilirim?",
    "goodbye": "Görüşmek üzere! İyi günler dilerim.",
    "konu_dışı": "Bu konuda yardımcı olamıyorum. Lütfen lojistikle ilgili bir soru sorun.",
}

# ──────────────────────────────────────────────
# 🚀 7. Uygulama
# ──────────────────────────────────────────────

def main():
    st.markdown('<h1 class="title">📦 Depo Asistanı Chatbotu</h1>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        model_choice = st.selectbox("Model Seç", ["OpenRouter Mistral", "Yerel Hugging Face Model"])
        if st.button("🗑️ Sohbet Geçmişini Temizle"):
            st.session_state.clear()
            st.rerun()

    # Session init
    messages = st.session_state.setdefault("messages", [])
    qa_chain = st.session_state.get("qa_chain")
    current_model = st.session_state.get("current_model")

    # Embedder / intent
    embedder = load_embedder()
    clf, enc = load_intent_classifier()
    ensure_faiss()

    # Chat UI
    for msg in messages:
        css = "user-message" if msg["role"] == "user" else "assistant-message"
        st.markdown(f'<div class="chat-message {css}"><strong>{"🧑‍💼 Siz" if msg["role"]=="user" else "🤖 Bot"}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Sorunuzu yazın…")
    if user_input:
        messages.append({"role": "user", "content": user_input})
        intent = enc.inverse_transform(clf.predict(embedder.encode([user_input])))[0]
        short = INTENT_SHORT.get(intent)
        if short:
            answer = short
        else:
            if current_model != model_choice or qa_chain is None:
                qa_chain = create_openrouter_qa_chain() if model_choice == "OpenRouter Mistral" else create_local_llm_qa_chain()
                st.session_state["qa_chain"] = qa_chain
                st.session_state["current_model"] = model_choice
            answer = qa_chain.run({"query": user_input})
        messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()
