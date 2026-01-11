import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Chatbot Hộ Chiếu Việt Nam", page_icon="🇻🇳")

# Lấy API Key từ Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ Chưa tìm thấy API Key trong Secrets!")
    st.stop()

# ------------------------------------------
# TỰ ĐỘNG TÌM TÊN MODEL ĐÚNG (SỬA LỖI 404)
# ------------------------------------------
@st.cache_resource
def find_correct_model_name():
    try:
        # Lấy danh sách tất cả model có hỗ trợ generateContent
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        # Ưu tiên tìm model Flash 1.5
        for name in available_models:
            if "1.5-flash" in name:
                return name
        # Nếu không thấy Flash, thử tìm bản Pro
        for name in available_models:
            if "pro" in name:
                return name
        return available_models[0]
    except Exception as e:
        # Nếu không liệt kê được, dùng tên mặc định phổ biến nhất
        return "models/gemini-1.5-flash"

AVAILABLE_MODEL = find_correct_model_name()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (RAG)
# ==========================================
@st.cache_resource
def init_vector_db():
    if not os.path.exists("TAI_LIEU_RB.json"):
        return None
    
    client = chromadb.PersistentClient(path="chroma_db_data")
    # Model embedding nhẹ cho Streamlit Cloud
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    try:
        collection = client.get_collection(name="RAG_passport", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="RAG_passport", embedding_function=emb_func)
        with open("TAI_LIEU_RB.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        collection.add(
            ids=[str(i) for i in range(len(data))],
            documents=[item["content_text"] for item in data],
            metadatas=[{"title": item["title"], "url": item["url"]} for item in data]
        )
    return collection

collection = init_vector_db()

# ==========================================
# 3. GIAO DIỆN & CHAT
# ==========================================
st.title("🇻🇳 Trợ lý ảo Hộ chiếu")
st.info(f"Hoạt động với model: `{AVAILABLE_MODEL}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            try:
                # Tìm kiếm trong database
                results = collection.query(query_texts=[user_input], n_results=2)
                context = "\n".join(results["documents"][0])
                
                # Gọi Gemini với tên model đã tìm thấy
                model = genai.GenerativeModel(model_name=AVAILABLE_MODEL)
                prompt = f"Ngữ cảnh: {context}\n\nCâu hỏi: {user_input}"
                response = model.generate_content(prompt)
                
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")
