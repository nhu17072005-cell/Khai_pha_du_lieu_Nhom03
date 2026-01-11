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
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    try:
        # Sử dụng collection mới để cập nhật metadata URL
        collection = client.get_collection(name="passport_official_v1", embedding_function=emb_func)
    except:
        collection = client.create_collection(name="passport_official_v1", embedding_function=emb_func)
        with open("TAI_LIEU_RB.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        collection.add(
            ids=[str(i) for i in range(len(data))],
            documents=[item["content_text"] for item in data],
            metadatas=[{"title": item["title"], "url": item["url"], "id": str(i)} for item in data]
        )
    return collection

collection = init_db()

# ==========================================
# 3. XỬ LÝ AI & TRÍCH DẪN NGUỒN
# ==========================================
def get_ai_response(user_query):
    # Tìm kiếm dữ liệu liên quan nhất
    results = collection.query(query_texts=[user_query], n_results=1)
    
    if not results["documents"][0]:
        return "Xin lỗi, tôi không tìm thấy thông tin này trong nguồn dữ liệu chính thức.", None, None

    context = results["documents"][0][0]
    meta = results["metadatas"][0][0]
    
    # Prompt yêu cầu trích dẫn rõ ràng theo block nội dung
    prompt = f"""Bạn là chuyên gia hướng dẫn dịch vụ công. 
Dựa vào tài liệu: {context}
Hãy trả lời câu hỏi: {user_query}
Yêu cầu: Trả lời chính xác, ngắn gọn. Tuyệt đối không tự chế link URL."""

    # Tìm model khả dụng (Flash hoặc Pro)
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model = "models/gemini-1.5-flash" if "models/gemini-1.5-flash" in available_models else available_models[0]
        
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(prompt)
        return response.text, meta['url'], meta['title']
    except Exception as e:
        return f"Lỗi kết nối AI: {str(e)}", None, None

# ==========================================
# 4. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
st.title("🇻🇳 Trợ lý ảo Dịch vụ công Chính thức")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập câu hỏi
user_input = st.chat_input("Nhập câu hỏi về thủ tục hành chính...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu từ nguồn dữ liệu chính thức..."):
            answer, source_url, source_title = get_ai_response(user_input)
            
            # Xây dựng phần hiển thị trích dẫn (như yêu cầu trong ảnh)
            formatted_answer = f"{answer}\n\n"
            if source_url:
                formatted_answer += f"**Trích dẫn nguồn:**\n"
                formatted_answer += f"- 📄 Tài liệu: *{source_title}*\n"
                formatted_answer += f"- 🔗 Link thực hiện dịch vụ: [Nhấn vào đây để truy cập]({source_url})"
            
            st.markdown(formatted_answer)
            st.session_state.messages.append({"role": "assistant", "content": formatted_answer})

with st.sidebar:
    st.header("Cam kết chất lượng")
    st.write("✅ Trả lời bằng tiếng Việt tự nhiên.")
    st.write("✅ Trích dẫn rõ nguồn gốc tài liệu.")
    st.write("✅ Dễ dàng kiểm chứng thông tin.")
    if st.button("Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()
