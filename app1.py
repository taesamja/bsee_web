import streamlit as st
import fitz  # PyMuPDF
from supabase import create_client, Client
import google.generativeai as genai
from openai import OpenAI
import json
import logging

# --- Supabase 클라이언트 초기화 ---
@st.cache_resource
def init_supabase_client() -> Client:
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except KeyError as e:
        st.error(f"'{e.args[0]}'를 secrets.toml 파일에 설정해주세요.")
        logging.error(f"Secrets 설정 오류: {e}")
        st.stop()

# --- Gemini 클라이언트 초기화 ---
@st.cache_resource
def init_gemini_model():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel("gemini-2.5-pro")
    except KeyError as e:
        st.error(f"'{e.args[0]}'를 secrets.toml 파일에 설정해주세요.")
        logging.error(f"Secrets 설정 오류: {e}")
        st.stop()
    except Exception as e:
        st.error("Gemini 모델 초기화에 실패했습니다. API 키가 유효한지 확인해주세요.")
        logging.error(f"Gemini 초기화 오류: {e}")
        st.stop()

# --- OpenAI 클라이언트 초기화 ---
@st.cache_resource
def init_openai_client():
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        return client
    except KeyError as e:
        st.error(f"'{e.args[0]}'를 secrets.toml 파일에 설정해주세요.")
        logging.error(f"Secrets 설정 오류: {e}")
        st.stop()

# --- PDF 텍스트 추출 ---
@st.cache_data
def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            text = "".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        st.error(f"PDF 텍스트 추출 중 오류가 발생했습니다: {e}")
        return ""

# --- 텍스트 분할 함수 (문자수 기준) ---
def split_text_into_chunks(text: str, max_chars: int = 1000) -> list:
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk += para + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# --- OpenAI 임베딩 생성 함수 ---
def get_embedding_openai(text: str):
    try:
        response = openai_client.embeddings.create(input=text, model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"임베딩 생성 오류: {e}")
        return None

# --- Supabase에 chunk + 임베딩 저장 ---
def save_vector_chunks(filename: str, text: str):
    chunks = split_text_into_chunks(text, max_chars=1000)
    for chunk in chunks:
        emb = get_embedding_openai(chunk)
        if emb:
            try:
                supabase.table("vector_chunks").insert({
                    "filename": filename,
                    "chunk_text": chunk,
                    "embedding": emb
                }).execute()
            except Exception as e:
                st.error(f"벡터 청크 저장 중 오류: {e}")

# --- PDF 텍스트와 vector_chunks 저장 통합 ---
def save_pdf_and_vectors(filename: str, content: str):
    try:
        supabase.table("pdf_texts").insert({"filename": filename, "content": content}).execute()
        save_vector_chunks(filename, content)
    except Exception as e:
        st.error(f"PDF 저장 및 벡터 업로드 오류: {e}")

# --- Supabase 벡터 유사도 검색 함수 ---
def query_similar_chunks(question: str, top_k=5) -> list:
    question_emb = get_embedding_openai(question)
    if not question_emb:
        return []

    try:
        response = supabase.rpc("match_chunks", {
            "query_embedding": question_emb,
            "match_count": top_k
        }).execute()

        if response.data is None or len(response.data) == 0:
            return []

        return [r["chunk_text"] for r in response.data]
    except Exception as e:
        st.error(f"벡터 검색 호출 중 오류 발생: {e}")
        return []

# --- 생성형 AI 답변 생성 함수 (RAG 방식) ---
def ask_question_with_rag(question: str, _model, _client, _api_choice: str) -> str:
    top_chunks = query_similar_chunks(question, top_k=5)
    if not top_chunks:
        return "관련된 내용을 찾지 못했습니다."
    context = "\n---\n".join(top_chunks)
    prompt = f"""다음은 관련 참고 문서 내용입니다.
{context}

아래 질문에 답변해 주세요:
{question}
"""
    try:
        if _api_choice == "Gemini":
            response = _model.generate_content(prompt)
            return response.text.strip()
        else:
            response = _client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API 호출 중 오류 발생: {e}"

# --- Gemini 요약 함수 ---
@st.cache_data
def summarize_text_gemini(text_chunk: str, _model) -> str:
    prompt = f"다음 텍스트를 간결하게 요약해줘:\n\n{text_chunk}"
    try:
        response = _model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"요약 중 오류 발생: {e}"

# --- OpenAI 요약 함수 ---
@st.cache_data
def summarize_text_openai(text_chunk: str, _client: OpenAI) -> str:
    prompt = f"다음 텍스트를 간결하게 요약해줘:\n\n{text_chunk}"
    try:
        response = _client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"요약 중 오류 발생: {e}"

# --- 퀴즈 생성 함수 (요약 텍스트 기반) ---
def generate_quiz_with_summary(text: str, _model, _client, _api_choice: str) -> str:
    chunks = split_text_into_chunks(text, max_chars=3000)
    summaries = []
    for chunk in chunks:
        if _api_choice == "Gemini":
            summary = summarize_text_gemini(chunk, _model)
        else:
            summary = summarize_text_openai(chunk, _client)
        summaries.append(summary)
    combined_summary = "\n".join(summaries)

    prompt = f"""
다음 텍스트를 바탕으로 객관식 문제 1개를 만들어줘.
반드시 아래 형식을 정확히 지켜서 응답해줘.

[형식]
문제: [문제 내용]
1. [보기 1]
2. [보기 2]
3. [보기 3]
4. [보기 4]
정답: [정답 보기의 내용]

[텍스트]
{combined_summary}
"""

    try:
        if _api_choice == "Gemini":
            response = _model.generate_content(prompt)
            return response.text.strip()
        else:
            response = _client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {e}"

# --- 퀴즈 저장 ---
def save_quiz(q: str, opts: list, ans: str):
    try:
        supabase.table("quiz_questions").insert({
            "question": q,
            "options": json.dumps(opts, ensure_ascii=False),
            "answer": ans
        }).execute()
        st.success("✅ 문제가 성공적으로 저장되었습니다.")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Supabase에 퀴즈 저장 중 오류 발생: {e}")

# --- 퀴즈 목록 조회 ---
@st.cache_data
def get_quizzes():
    try:
        response = supabase.table("quiz_questions").select("*").execute()
        return response.data
    except Exception as e:
        st.error(f"퀴즈 목록을 불러오는 중 오류 발생: {e}")
        return []

# --- Supabase 초기화 ---
supabase = init_supabase_client()
gemini_model = init_gemini_model()
openai_client = init_openai_client()

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="PDF 기반 Q&A RAG 시스템", layout="wide")
st.title("📚 회로용어 EASY AI - 내 손안의 맞춤형 학습 코치")

api_choice = st.sidebar.radio("AI 모델 선택", ("Gemini", "OpenAI"))
menu = st.sidebar.selectbox("메뉴 선택", ["📄 PDF 업로드", "❓ 질의응답", "📝 시험문제 출제", "📑 문제 보기"])

if menu == "📄 PDF 업로드":
    st.subheader("📄 PDF 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 업로드해주세요.", type="pdf")

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        pdf_text = extract_text_from_pdf(file_bytes)

        if f"processed_{uploaded_file.name}" not in st.session_state:
            with st.spinner("PDF 텍스트를 분석하고 벡터 임베딩을 저장하는 중..."):
                save_pdf_and_vectors(uploaded_file.name, pdf_text)
            st.session_state[f"processed_{uploaded_file.name}"] = True
            st.success("✅ PDF 텍스트와 벡터 임베딩 저장 완료!")
        else:
            st.info("ℹ️ 이전에 처리된 PDF입니다. 캐시된 데이터를 사용합니다.")

    st.divider()

elif menu == "❓ 질의응답":
    st.subheader("❓ 질의응답")
    try:
        pdf_list_response = supabase.table("pdf_texts").select("id, filename").execute()
        pdf_list = pdf_list_response.data
    except Exception as e:
        st.error(f"저장된 PDF 목록을 불러오는 중 오류 발생: {e}")
        pdf_list = []

    if pdf_list:
        pdf_choice = st.selectbox("문서 선택", pdf_list, format_func=lambda x: x['filename'])
        if pdf_choice:
            try:
                selected_pdf_content_response = supabase.table("pdf_texts").select("content").eq("id", pdf_choice['id']).single().execute()
                selected_pdf_content = selected_pdf_content_response.data['content']
            except Exception as e:
                st.error(f"선택된 PDF 내용을 불러오는 중 오류 발생: {e}")
                selected_pdf_content = ""

            user_q = st.text_input("PDF 내용에 대해 질문해보세요:", placeholder="예: 이 문서의 핵심 내용은 무엇인가요?")
            if user_q and selected_pdf_content:
                with st.spinner("답변을 생성하는 중..."):
                    answer = ask_question_with_rag(
                        user_q,
                        _model=gemini_model,
                        _client=openai_client,
                        _api_choice=api_choice)
                    st.info(f"💬 답변: {answer}")
    else:
        st.info("먼저 '📄 PDF 업로드' 메뉴에서 PDF 파일을 업로드해주세요.")
    st.divider()

elif menu == "📝 시험문제 출제":
    st.subheader("🧠 퀴즈 생성 및 저장")
    try:
        pdf_list_response = supabase.table("pdf_texts").select("id, filename").execute()
        pdf_list = pdf_list_response.data
    except Exception as e:
        st.error(f"저장된 PDF 목록을 불러오는 중 오류 발생: {e}")
        pdf_list = []

    if pdf_list:
        pdf_choice = st.selectbox("출제할 문서 선택", pdf_list, format_func=lambda x: x['filename'])
        if st.button("객관식 문제 생성하기", type="primary"):
            if pdf_choice:
                try:
                    selected_pdf_content_response = supabase.table("pdf_texts").select("content").eq("id", pdf_choice['id']).single().execute()
                    selected_pdf_content = selected_pdf_content_response.data['content']
                except Exception as e:
                    st.error(f"선택된 PDF 내용을 불러오는 중 오류 발생: {e}")
                    selected_pdf_content = ""

                if selected_pdf_content:
                    with st.spinner("퀴즈를 생성하는 중..."):
                        quiz_text = generate_quiz_with_summary(
                            selected_pdf_content,
                            _model=gemini_model,
                            _client=openai_client,
                            _api_choice=api_choice)
                        try:
                            lines = [line.strip() for line in quiz_text.splitlines() if line.strip()]
                            question = ""
                            options = []
                            answer = ""

                            for line in lines:
                                if line.startswith("문제:"):
                                    question = line.replace("문제:", "").strip()
                                elif line.startswith(("1.", "2.", "3.", "4.")):
                                    options.append(line[2:].strip())
                                elif line.startswith("정답:"):
                                    answer = line.replace("정답:", "").strip()

                            if question and len(options) == 4 and answer:
                                st.write("**생성된 문제 미리보기**")
                                st.code(quiz_text, language='text')
                                save_quiz(question, options, answer)
                            else:
                                st.error("❌ 문제 파싱 실패. 선택한 AI가 생성한 퀴즈 형식을 확인하세요.")
                                st.code(quiz_text, language='text')

                        except Exception as e:
                            st.error(f"❌ 문제 저장 중 오류 발생: {e}")
                            st.write("AI가 생성한 원본 텍스트:")
                            st.code(quiz_text, language='text')
                else:
                    st.info("먼저 문서를 선택해주세요.")
    else:
        st.info("먼저 '📄 PDF 업로드' 메뉴에서 PDF 파일을 업로드해주세요.")
    st.divider()

elif menu == "📑 문제 보기":
    st.subheader("📋 저장된 문제 목록")
    quiz_data = get_quizzes()
    if quiz_data:
        for q in reversed(quiz_data):
            st.markdown(f"**Q. {q['question']}**")
            try:
                options_list = json.loads(q["options"]) if isinstance(q["options"], str) else q["options"]
            except json.JSONDecodeError:
                options_list = []

            correct_answer_text = q['answer'].strip()
            for opt in options_list:
                if opt.strip() == correct_answer_text:
                    st.markdown(f"- {opt} ✅")
                else:
                    st.markdown(f"- {opt}")
            st.markdown("---")
    else:
        st.info("현재 저장된 문제가 없습니다.")
