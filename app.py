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
    st.error("❌ Vui lòng dán API Key vào mục Secrets của Streamlit!")
    st.stop()

# ĐỊNH DANH MODEL CHUẨN (Ép dùng Flash để có Quota cao nhất)
MODEL_NAME = "gemini-1.5-flash"

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (RAG)
# ==========================================
@st.cache_resource
def init_db():
    if not os.path.exists("TAI_LIEU_RB.json"):
        return None
    
    # Khởi tạo Vector DB nhẹ
    client = chromadb.PersistentClient(path="chroma_db_data")
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    try:
        collection = client.get_collection(name="passport_rag", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="passport_rag", embedding_function=emb_func)
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
# 3. HÀM XỬ LÝ PHẢN HỒI (TỐI ƯU QUOTA)
# ==========================================
def get_ai_response(user_query):
    # Tìm kiếm 1 đoạn văn duy nhất để tiết kiệm Token đầu vào
    results = collection.query(query_texts=[user_query], n_results=1)
    context = results["documents"][0][0] if results["documents"] else "Không tìm thấy dữ liệu."

    # Prompt tối giản để tiết kiệm hạn mức
    prompt = f"Ngữ cảnh: {context}\nTrả lời ngắn gọn câu hỏi: {user_query}"
    
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

# ==========================================
# 4. GIAO DIỆN CHAT
# ==========================================
st.title("🇻🇳 Trợ lý Hộ chiếu Việt Nam")
st.caption(f"Đang sử dụng hệ thống: {MODEL_NAME} (Free Tier)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Hỏi về lệ phí, thủ tục...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            try:
                answer = get_ai_response(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.warning("⚠️ Bạn đã dùng hết lượt miễn phí trong phút này. Vui lòng đợi 30-60 giây rồi thử lại.")
                elif "404" in error_msg:
                    st.error("❌ Model hiện tại không khả dụng. Vui lòng kiểm tra lại API Key.")
                else:
                    st.error(f"Lỗi hệ thống: {error_msg}")

# Sidebar
with st.sidebar:
    st.header("Lưu ý")
    st.write("- Chỉ hỏi về thủ tục hộ chiếu.")
    st.write("- Nếu bị lỗi quá tải, hãy chờ 1 phút.")
    if st.button("Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()
