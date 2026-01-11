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

# Lấy API Key từ Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ Vui lòng dán API Key vào mục Secrets của Streamlit Cloud!")
    st.stop()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (RAG)
# ==========================================
@st.cache_resource
def init_db():
    if not os.path.exists("TAI_LIEU_RB.json"):
        return None
    
    # Khởi tạo Vector DB
    client = chromadb.PersistentClient(path="chroma_db_data")
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    try:
        collection = client.get_collection(name="passport_rag_v2", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="passport_rag_v2", embedding_function=emb_func)
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
# 3. HÀM GỌI AI (TỰ ĐỘNG SỬA LỖI MODEL)
# ==========================================
def get_ai_response(user_query):
    # 1. Tra cứu dữ liệu (Chỉ lấy 1 đoạn để tiết kiệm Quota)
    results = collection.query(query_texts=[user_query], n_results=1)
    context = results["documents"][0][0] if results["documents"] else "Không tìm thấy dữ liệu."

    prompt = f"Dựa vào thông tin: {context}. Hãy trả lời ngắn gọn câu hỏi: {user_query}"
    
    # 2. Danh sách các tên model có thể hoạt động (để tránh lỗi 404)
    model_names = ["models/gemini-1.5-flash", "gemini-1.5-flash"]
    
    last_error = ""
    for name in model_names:
        try:
            model = genai.GenerativeModel(name)
            response = model.generate_content(prompt)
            return response.text, name
        except Exception as e:
            last_error = str(e)
            continue # Thử tên model tiếp theo
            
    # Nếu tất cả đều lỗi, ném lỗi ra ngoài
    raise Exception(last_error)

# ==========================================
# 4. GIAO DIỆN CHAT
# ==========================================
st.title("🇻🇳 Trợ lý Hộ chiếu Việt Nam")
st.caption("Dữ liệu tra cứu thủ tục chính thức")

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
                answer, used_model = get_ai_response(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                # Hiển thị model đang chạy để theo dõi
                st.info(f"💡 Phản hồi từ: {used_model}", icon="✅")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.warning("⚠️ Đang quá tải. Vui lòng chờ 30-60 giây rồi thử lại.")
                elif "404" in error_msg:
                    st.error("❌ Model hiện tại không khả dụng. Có thể do API Key hoặc khu vực.")
                else:
                    st.error(f"Lỗi: {error_msg}")

with st.sidebar:
    st.header("Thông tin")
    st.write("- Hệ thống tự động chọn model ổn định nhất.")
    if st.button("Xóa lịch sử"):
        st.session_state.messages = []
        st.rerun()
