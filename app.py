import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & BẢO MẬT
# ==========================================
st.set_page_config(
    page_title="Chatbot Hộ Chiếu Việt Nam",
    page_icon="🇻🇳",
    layout="centered"
)

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ Chưa tìm thấy API Key trong Secrets!")
    st.stop()

# ------------------------------------------
@st.cache_resource
def get_safe_model_name():
    try:
        # Liệt kê model để kiểm tra tính khả dụng
        models = [m.name for m in genai.list_models()]
        for m in models:
            if "1.5-flash" in m: return m
        return "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

AVAILABLE_MODEL = get_safe_model_name()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & VECTOR DB
# ==========================================
JSON_FILE = "TAI_LIEU_RB.json"
CHROMA_DB_PATH = "chroma_db_data"

@st.cache_resource
def get_embedding_function():
    # Sử dụng model nhỏ để không tốn RAM của Streamlit
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
        collection = client.get_collection(name="RAG_procedure", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="RAG_procedure", embedding_function=emb_func)
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
# 3. LOGIC XỬ LÝ CHAT (RAG)
# ==========================================
def get_chatbot_response(user_query):
    # Tìm kiếm 2 đoạn tin quan trọng nhất (giảm xuống 2 để tiết kiệm Token)
    results = collection.query(query_texts=[user_query], n_results=2)
    
    context_text = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_text += f"\n[Nguồn: {meta['title']}]\n{doc}\n---\n"

    full_prompt = f"""Bạn là chuyên gia tư vấn hộ chiếu Việt Nam.
Dựa vào ngữ cảnh dưới đây, hãy trả lời câu hỏi ngắn gọn, chính xác.
Nếu thông tin không có, hãy nói bạn không biết.

NGỮ CẢNH:
{context_text}

CÂU HỎI: {user_query}"""

    model = genai.GenerativeModel(model_name=AVAILABLE_MODEL)
    # Cấu hình giảm token đầu ra để tiết kiệm quota
    response = model.generate_content(full_prompt)
    return response.text

# ==========================================
# 4. GIAO DIỆN NGƯỜI DÙNG
# ==========================================
st.title("🇻🇳 Trợ lý ảo Hộ chiếu (Tối ưu Quota)")
st.info(f"Hoạt động với model: `{AVAILABLE_MODEL}`")

if collection is None:
    st.error(f" Không thấy file `{JSON_FILE}`!")
    st.stop()

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
        with st.spinner("Đang tìm lời giải..."):
            try:
                answer = get_chatbot_response(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                if "429" in str(e):
                    st.error("Hệ thống đang quá tải (Hết lượt dùng miễn phí). Vui lòng thử lại sau 1 phút.")
                else:
                    st.error(f"Lỗi: {str(e)}")

with st.sidebar:
    st.markdown("### Hướng dẫn")
    st.write("Nếu gặp lỗi 429, vui lòng chờ khoảng 60 giây trước khi hỏi câu tiếp theo.")
    if st.button("Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()
