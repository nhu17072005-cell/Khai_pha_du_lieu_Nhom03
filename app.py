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
    api_key = "AIzaSyCzcZwCm4cycmjT2Q1biZNYDfbI5sh9Cr4"

genai.configure(api_key=api_key)

# Cấu hình file và model
JSON_FILE = "TAI_LIEU_RB.json" 
CHROMA_DB_PATH = "chroma_db_data"
COLLECTION_NAME = "RAG_procedure"
GEMINI_MODEL_NAME = "gemini-1.5-flash" 

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & EMBEDDING
# ==========================================
@st.cache_resource
def get_embedding_function():
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
# 3. LOGIC CHATBOT (RAG)
# ==========================================
def get_chatbot_response(user_query):
    # 1. Tìm kiếm trong Vector DB
    results = collection.query(query_texts=[user_query], n_results=3)
    
    context_text = ""
    # Duyệt qua kết quả tìm kiếm để xây dựng ngữ cảnh
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_text += f"\n[Nguồn: {meta['title']}]\n{doc}\nLink: {meta['url']}\n---\n"

    # 2. Tạo Prompt gửi cho AI
    full_prompt = f"""Bạn là chuyên gia hướng dẫn thủ tục hành chính tại Việt Nam. 
Hãy trả lời câu hỏi dựa trên Context dưới đây một cách lịch sự, chính xác.
Nếu thông tin không có trong Context, hãy hướng dẫn người dùng liên hệ Cổng Dịch vụ công hoặc Cơ quan Công an.

CONTEXT:
{context_text}

CÂU HỎI: {user_query}
"""

    # 3. Gọi Gemini (Đã sửa lỗi đóng ngoặc dòng 92)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
    response = model.generate_content(full_prompt)
    return response.text

# ==========================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
st.title("🇻🇳 Trợ lý ảo Thủ tục Hộ chiếu")
st.caption("Dữ liệu tra cứu từ Cổng Dịch vụ công Quốc gia")

if collection is None:
    st.error(f"❌ Không tìm thấy file `{JSON_FILE}`. Vui lòng kiểm tra lại trên GitHub.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập liệu từ người dùng
user_input = st.chat_input("Hỏi về thủ tục làm hộ chiếu...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu dữ liệu..."):
            try:
                answer = get_chatbot_response(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Emblem_of_Vietnam.svg/512px-Emblem_of_Vietnam.svg.png", width=80)
    st.header("Hướng dẫn")
    st.write("Sử dụng dữ liệu hành chính công để giải đáp các thắc mắc về hộ chiếu.")
    if st.button("Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.rerun()
