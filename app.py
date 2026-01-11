import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
st.set_page_config(page_title="Hỗ trợ Hộ chiếu VN", page_icon="🇻🇳", layout="wide")

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ Thiếu API Key trong Secrets!")
    st.stop()

# ==========================================
# 2. KHỞI TẠO DỮ LIỆU (RAG)
# ==========================================
@st.cache_resource
def init_db():
    if not os.path.exists("TAI_LIEU_RB.json"):
        return None
    
    client = chromadb.PersistentClient(path="chroma_db_data")
    
    # Sử dụng model embedding mặc định của SentenceTransformer
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    try:
        # Xóa hoặc đổi tên collection nếu bạn thay đổi cấu trúc dữ liệu
        collection = client.get_or_create_collection(name="passport_official_v4", embedding_function=emb_func)
        
        # Kiểm tra nếu collection còn trống mới nạp dữ liệu
        if collection.count() == 0:
            with open("TAI_LIEU_RB.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # SỬA LỖI TẠI ĐÂY: Dùng enumerate để có cả index (i) và nội dung (item)
            documents = [item["content_text"] for item in data]
            metadatas = [{"title": item["title"], "url": item["url"], "id": str(i)} for i, item in enumerate(data)]
            ids = [str(i) for i in range(len(data))]
            
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
    except Exception as e:
        st.error(f"Lỗi khởi tạo Database: {e}")
        return None
        
    return collection

collection = init_db()

# ==========================================
# 3. XỬ LÝ AI
# ==========================================
def get_ai_response(user_query):
    if collection is None:
        return "Dữ liệu chưa được khởi tạo.", None, None

    # Tìm kiếm dữ liệu
    results = collection.query(query_texts=[user_query], n_results=1)
    
    if not results["documents"] or not results["documents"][0]:
        return "Không tìm thấy thông tin phù hợp.", None, None

    context = results["documents"][0][0]
    meta = results["metadatas"][0][0]
    
    prompt = f"Dữ liệu: {context}\n\nCâu hỏi: {user_query}\nTrả lời ngắn gọn, chính xác."

    try:
        # Thử các model phổ biến
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text, meta['url'], meta['title']
    except Exception:
        return "Lỗi kết nối AI. Vui lòng thử lại sau.", None, None

# ==========================================
# 4. GIAO DIỆN
# ==========================================
st.title("🇻🇳 Trợ lý ảo Hộ chiếu")

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
        answer, url, title = get_ai_response(user_input)
        
        full_res = f"{answer}\n\n
