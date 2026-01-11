import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Hỗ trợ Hộ chiếu VN", page_icon="🇻🇳")

# Kiểm tra Key trong Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ Thiếu API Key trong Secrets!")
    st.stop()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (RAG)
# ==========================================
@st.cache_resource
def init_db():
    if not os.path.exists("TAI_LIEU_RB.json"):
        return None
    client = chromadb.PersistentClient(path="chroma_db_data")
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    try:
        collection = client.get_collection(name="passport_rag_final", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="passport_rag_final", embedding_function=emb_func)
        with open("TAI_LIEU_RB.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        collection.add(
            ids=[str(i) for i in range(len(data))],
            documents=[item["content_text"] for item in data],
            metadatas=[{"title": item["title"]} for item in data]
        )
    return collection

collection = init_db()

# ==========================================
# 3. CHIẾN THUẬT TỰ ĐỘNG THỬ MODEL (MODEL CYCLING)
# ==========================================
def generate_with_fallback(prompt):
    # Bước 1: Lấy danh sách thực tế các model mà KEY này dùng được
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    except:
        available_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]

    # Bước 2: Thử từng model trong danh sách
    errors = []
    for model_name in available_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, model_name
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")
            continue
            
    # Nếu tất cả đều thất bại
    st.error("Tất cả các model đều không phản hồi. Chi tiết lỗi:")
    for err in errors: st.write(err)
    return None, None

# ==========================================
# 4. GIAO DIỆN
# ==========================================
st.title("🇻🇳 Trợ lý Hộ chiếu Việt Nam")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Nhập câu hỏi...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang kết nối AI..."):
            # Tìm kiếm ngữ cảnh
            results = collection.query(query_texts=[user_input], n_results=1)
            context = results["documents"][0][0] if results["documents"] else ""
            
            full_prompt = f"Dữ liệu: {context}\n\nCâu hỏi: {user_input}\nTrả lời ngắn gọn bằng tiếng Việt."
            
            # Gọi hàm Fallback
            answer, success_model = generate_with_fallback(full_prompt)
            
            if answer:
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.caption(f"✅ Đã chạy thành công trên: `{success_model}`")
