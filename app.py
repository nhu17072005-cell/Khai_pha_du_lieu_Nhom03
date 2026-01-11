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

# Lấy API Key từ Secrets của Streamlit Cloud (Bảo mật)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    # Key dự phòng (Nếu chạy local thì điền trực tiếp ở đây hoặc dùng file secrets.toml)
    api_key = "AIzaSyCzcZwCm4cycmjT2Q1biZNYDfbI5sh9Cr4"

genai.configure(api_key=api_key)

# Cấu hình đường dẫn file
JSON_FILE = "TAI_LIEU_RB.json" 
CHROMA_DB_PATH = "chroma_db_data"
COLLECTION_NAME = "RAG_procedure"
GEMINI_MODEL = "gemini-1.5-flash" 

# ==========================================
# 2. XỬ LÝ DỮ LIỆU & EMBEDDING
# ==========================================
@st.cache_resource
def get_embedding_function():
    # Sử dụng model đa ngôn ngữ nhẹ để tối ưu RAM trên Cloud
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
        # Thử kết nối collection cũ
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_func)
    except:
        # Nếu chưa có thì tạo mới và nạp dữ liệu từ file JSON
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

# Khởi tạo database
collection = init_vector_db()

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
st.title("🇻🇳 Trợ lý ảo Thủ tục Hộ chiếu")
st.markdown("---")

if collection is None:
    st.error(f"❌ Không tìm thấy file `{JSON_FILE}`. Hãy đảm bảo bạn đã upload file này lên GitHub.")
    st.stop()

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn cũ
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Hàm xử lý truy vấn RAG
def get_chatbot_response(user_query):
    # 1. Tìm kiếm trong Vector DB
    results = collection.query(query_texts=[user_query], n_results=3)
    
    # 2. Xây dựng ngữ cảnh (Context)
    context_text = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_text += f"\n[Nguồn: {meta['title']}]\n{doc}\nLink: {meta['url']}\n---"

    # 3. Tạo Prompt cho Gemini
    full_prompt = f"""Bạn là một chuyên gia hướng dẫn thủ tục hành chính tại Việt Nam. 
Hãy sử dụng thông tin dưới đây để trả lời câu hỏi một cách chính xác, lịch sự.
Nếu thông tin không có trong Context, hãy nói bạn không biết và khuyên người dùng tra cứu thêm tại Cổng Dịch vụ công.

CONTEXT:
{context_text}

CÂU HỎI: {user_query}
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(full_prompt)
    return response.text

# Ô nhập liệu chat
user_input = st.chat_input("Hỏi về làm hộ chiếu, cấp đổi, lệ phí...")

if user_input:
    # Lưu và hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Xử lý phản hồi từ chatbot
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu dữ liệu..."):
            try:
                answer = get_chatbot_response(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")

# Thanh bên (Sidebar)
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Emblem_of_Vietnam.svg/512px-Emblem_of_Vietnam.svg.png", width=80)
    st.header("Thông tin")
    st.info("Chatbot sử dụng công nghệ RAG kết hợp Google Gemini để tư vấn thủ tục hành chính.")
    if st.button("Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.rerun()
