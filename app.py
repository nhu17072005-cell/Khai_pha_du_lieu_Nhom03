import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(
    page_title="Chatbot Hộ Chiếu Việt Nam",
    page_icon="🇻🇳",
    layout="centered"
)

# Lấy API Key từ Secrets của Streamlit Cloud
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    # Nếu chạy local mà chưa có secrets.toml thì dán tạm ở đây để test
    api_key = "AIzaSyCzcZwCm4cycmjT2Q1biZNYDfbI5sh9Cr4"

genai.configure(api_key=api_key)

# Cấu hình đường dẫn file và model
JSON_FILE = "TAI_LIEU_RB.json" 
CHROMA_DB_PATH = "chroma_db_data"
COLLECTION_NAME = "RAG_procedure"

# Sửa lỗi 404 bằng cách dùng tên model chuẩn và thêm bọc kiểm tra
GEMINI_MODEL_NAME = "gemini-1.5-flash" 

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & EMBEDDING
# ==========================================
@st.cache_resource
def get_embedding_function():
    # Model đa ngôn ngữ nhẹ, phù hợp RAM 1GB của Streamlit Cloud
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

@st.cache_resource
def init_vector_db():
    if not os.path.exists(JSON_FILE):
        return None
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    emb_func = get_embedding_function()
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_func)
    except:
        collection = client.create_collection(name=COLLECTION_NAME, embedding_function=emb_func)
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        collection.add(
            ids=[str(i) for i in range(len(data))],
            documents=[item["content_text"] for item in data],
            metadatas=[
                {"url": item["url"], "title": item["title"], "hierarchy": item["hierarchy"]}
                for item in data
            ]
        )
    return collection

collection = init_vector_db()

# ==========================================
# 3. LOGIC CHATBOT
# ==========================================
def get_chatbot_response(user_query):
    # 1. Tìm kiếm thông tin trong Vector DB
    results = collection.query(query_texts=[user_query], n_results=3)
    
    context_text = ""
    for doc, meta in zip
