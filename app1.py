import streamlit as st
import fitz  # PyMuPDF
import hashlib
from supabase import create_client, Client
import google.generativeai as genai
from openai import OpenAI
import json
import logging
import datetime
import pytz
import pandas as pd

# --- 스타일링 및 CSS ---
st.markdown("""
<style>
    .big-font { font-size: 2.8em !important; font-weight: bold; color: #2196F3; margin-bottom: 0.3em; }
    .card { background: #ffffff; color: #111111; padding: 1.2em 1.5em; border-radius: 14px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 1em; transition: background 0.2s ease; }
    .card:hover { background: #e3f2fd; cursor: pointer; }
    .divider { border-top: 1px solid #ddd; margin: 2.5em 0 2em 0; }
    .stButton>button { font-weight: bold; color: white; background-color: #2196F3; border-radius: 8px; padding: 0.45em 1em; transition: background-color 0.3s ease; min-width: 40px; }
    .stButton>button:hover { background-color: #1976D2; }
    .stTextInput>div>input { border-radius: 10px !important; border: 1.8px solid #90caf9 !important; padding: 0.5em 1em !important; font-size: 1.1rem !important; }
    [data-baseweb="radio"] label, .css-1okebmr span, .css-1r6slb0 {
        font-size: 1.1rem !important;
        font-weight: 600;
    }
    /* 라디오 버튼 간격 확대 */
    .stRadio > div[role='radiogroup'] > label {
        margin-bottom: 1rem !important;
        margin-right: 2.2rem !important;
        padding: 0.25em 0.75em !important;
        min-width: 90px;
        cursor: pointer;
    }
    /* 페이지네이션 컨테이너 */
    .pagination-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.2rem;
        margin-top: 1rem;
    }
    /* 페이지네이션 버튼 커스텀 */
    .stButton > button {
        border: 2px solid #2196F3 !important;
        background-color: white !important;
        color: #2196F3 !important;
        font-weight: 600 !important;
        min-width: 36px !important;
        padding: 0.35em 0.85em !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }
    .stButton > button:hover:not(:disabled) {
        background-color: #e3f2fd !important;
        color: #1976D2 !important;
    }
    .stButton > button:disabled {
        background-color: #2196F3 !important;
        color: white !important;
        cursor: default !important;
        border-color: #2196F3 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 초기화 함수 및 API 클라이언트 설정 ---

@st.cache_resource
def init_supabase_client() -> Client:
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except KeyError as e:
        st.error(f"'{e.args[0]}' secrets.toml 파일에 설정해 주세요.")
        logging.error(f"Secrets 오류: {e}")
        st.stop()

@st.cache_resource
def init_gemini_model():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel("gemini-2.5-flash-lite")
    except Exception as e:
        st.error("Gemini 모델 초기화 실패. API 키 및 연결을 확인해 주세요.")
        logging.error(f"Gemini 초기화 오류: {e}")
        st.stop()

@st.cache_resource
def init_openai_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except KeyError as e:
        st.error(f"'{e.args[0]}' secrets.toml 파일에 설정해 주세요.")
        logging.error(f"Secrets 오류: {e}")
        st.stop()


# --- PDF 텍스트 추출 함수, 저장 관련 함수 등 ---

@st.cache_data
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"PDF 텍스트 추출 오류: {e}")
        return ""

def get_pdf_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def save_pdf_text(filename: str, content: str, content_hash: str):
    try:
        supabase.table("pdf_texts").insert({
            "filename": filename,
            "content": content,
            "content_hash": content_hash,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        st.error(f"PDF 텍스트 저장 오류: {e}")

def check_pdf_exists(filename: str, content_hash: str) -> bool:
    try:
        res = supabase.table("pdf_texts").select("id").or_(
            f"filename.eq.{filename},content_hash.eq.{content_hash}"
        ).limit(1).execute()
        return bool(res.data)
    except Exception as e:
        st.error(f"PDF 중복 체크 오류: {e}")
        return False


# --- AI 질문 답변 및 퀴즈 생성 함수 ---

@st.cache_data
def ask_gemini_with_pdf(question: str, _gemini_model, filename: str) -> str:
    try:
        res = supabase.table("pdf_texts").select("content").eq("filename", filename).single().execute()
        if not res.data or not res.data.get("content"):
            return "관련 내용을 찾지 못했습니다."
        context = res.data["content"]
        prompt = f"""다음은 참고 문서 내용입니다:
{context}

위 내용을 참고하여 아래 질문에 답변해 주세요:

질문: {question}
"""
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API 오류: {e}"

@st.cache_data
def ask_openai_simple(question: str, _openai_client) -> str:
    try:
        prompt = f"다음 질문에 대해 간결하고 정확하게 답변해 주세요:\n\n질문: {question}"
        response = _openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API 오류: {e}"

@st.cache_data
def summarize_text_with_ai(text_chunk: str, _gemini_model, _api_choice: str) -> str:
    prompt = f"다음 텍스트를 간결하고 명확하게 요약해줘:\n\n{text_chunk}"
    try:
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"요약 오류: {e}"

def generate_quiz_with_summary(text: str, _gemini_model, _api_choice: str) -> str:
    chunk_size = 3000
    start = 0
    summaries = []
    while start < len(text):
        chunk = text[start: start + chunk_size]
        summary = summarize_text_with_ai(chunk, _gemini_model, _api_choice)
        summaries.append(summary)
        start += chunk_size
    combined_summary = "\n".join(summaries)
    prompt = f"""
다음 텍스트를 바탕으로 객관식 문제 1개를 만들어줘.
반드시 아래 형식을 지켜서 응답해줘.

[형식]
문제: [문제 내용]
1. [보기 1]
2. [보기 2]
3. [보기 3]
4. [보기 4]
정답: [정답 보기의 내용]
해설: [문제 풀이 간단 설명]

[텍스트]
{combined_summary}
"""
    try:
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"API 호출 오류: {e}"

def parse_and_save_quiz(quiz_text: str, _unit: str = ""):
    lines = [line.strip() for line in quiz_text.splitlines() if line.strip()]
    question = ""
    options = []
    answer = ""
    explanation = ""

    for line in lines:
        if line.startswith("문제:"):
            question = line.replace("문제:", "").strip()
        elif line.startswith(("1.", "2.", "3.", "4.")):
            options.append(line[2:].strip())
        elif line.startswith("정답:"):
            answer = line.replace("정답:", "").strip()
        elif line.startswith("해설:"):
            explanation = line.replace("해설:", "").strip()

    matched_answer = None
    answer_lower = answer.lower()
    for opt in options:
        if opt.lower() == answer_lower:
            matched_answer = opt
            break
    if matched_answer is None:
        for opt in options:
            if answer_lower in opt.lower() or opt.lower() in answer_lower:
                matched_answer = opt
                break
    fixed_answer = matched_answer if matched_answer is not None else answer

    if question and len(options) == 4 and fixed_answer:
        save_quiz(question, options, fixed_answer, explanation, _unit)
        st.success("✅ 퀴즈가 성공적으로 저장되었습니다 (후처리 적용됨).")
        return True
    else:
        st.error("❌ 퀴즈 파싱 실패 또는 데이터 부족입니다.")
        st.code(quiz_text, language="text")
        return False

def save_quiz(question: str, options: list, answer: str, explanation: str = "", unit: str = ""):
    try:
        supabase.table("quiz_questions").insert({
            "question": question,
            "options": json.dumps(options, ensure_ascii=False),
            "answer": answer,
            "explanation": explanation,
            "unit": unit,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }).execute()
        st.cache_data.clear()
    except Exception as e:
        st.error(f"퀴즈 저장 오류: {e}")

@st.cache_data
def get_quizzes():
    try:
        response = supabase.table("quiz_questions").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"퀴즈 목록 조회 오류: {e}")
        return []


# --- 초기화 ---
supabase = init_supabase_client()
gemini_model = init_gemini_model()
openai_client = init_openai_client()


# --- 페이지네이션 렌더링 함수 ---
def render_styled_pagination(current_page, total_pages, key_prefix="pagination"):
    max_buttons = 10
    start_page = max(1, current_page - 4)
    end_page = min(total_pages, start_page + max_buttons - 1)
    start_page = max(1, end_page - max_buttons + 1)
    btn_count = end_page - start_page + 1

    cols = st.columns(btn_count, gap="small")
    page_clicked = None

    for i, page_num in enumerate(range(start_page, end_page + 1)):
        disabled = (page_num == current_page)
        with cols[i]:
            if st.button(str(page_num), key=f"{key_prefix}_{page_num}", disabled=disabled):
                page_clicked = page_num
    return page_clicked


# --- Streamlit UI & 세션 상태 초기화 ---

st.set_page_config(page_title="PDF 기반 Q&A 시스템", layout="wide")
st.markdown('<p class="big-font">📚 EASY AI 맞춤 학습 도우미</p>', unsafe_allow_html=True)
st.info("AI 기반 PDF 학습자료 관리, Q&A, 자동 시험문제 생성이 여기에!")

# 초기 세션 상태 값 설정
if "api_choice" not in st.session_state:
    st.session_state.api_choice = "Gemini"
if "menu" not in st.session_state:
    st.session_state.menu = "📄 학습자료 PDF 업로드"
if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False
if "question_submitted_simple" not in st.session_state:
    st.session_state.question_submitted_simple = False
if "generated_quiz_raw" not in st.session_state:
    st.session_state.generated_quiz_raw = ""

# 콜백 함수
def api_choice_changed():
    st.session_state.menu = "📄 학습자료 PDF 업로드"

def menu_changed():
    pass


# 사이드바 위젯, key와 on_change 지정 - 상태는 별도 할당 하지 않음
st.sidebar.radio("AI 모델 선택", ("Gemini", "OpenAI"),
                 index=0 if st.session_state.api_choice == "Gemini" else 1,
                 key="api_choice", on_change=api_choice_changed)

menu_names = ["📄 학습자료 PDF 업로드", "❓ 질의응답", "📝 시험문제 출제", "📑 문제 보기"]
st.sidebar.radio("메뉴 선택", menu_names,
                 index=menu_names.index(st.session_state.menu),
                 key="menu", on_change=menu_changed)

api_choice = st.session_state.api_choice
selected_menu = st.session_state.menu


# --- 메뉴 분기 ---

if selected_menu == "📄 학습자료 PDF 업로드":
    st.subheader("📄 학습자료 PDF 업로드")
    uploaded_file = st.file_uploader("학습자료 PDF 파일을 업로드해주세요. (2MB 이하) - Drag & Drop 가능", type="pdf")
    if uploaded_file:
        max_mb = 2
        if uploaded_file.size > max_mb * 1024 * 1024:
            st.error(f"⚠️ 파일 크기가 {max_mb}MB를 초과했습니다. 다른 파일을 선택해주세요.")
        else:
            file_bytes = uploaded_file.getvalue()
            pdf_text = extract_text_from_pdf(file_bytes)
            pdf_hash = get_pdf_content_hash(pdf_text)
            if check_pdf_exists(uploaded_file.name, pdf_hash):
                st.error("❌ 동일한 파일(이름 또는 내용)이 이미 업로드되어 있습니다.")
            else:
                save_pdf_text(uploaded_file.name, pdf_text, pdf_hash)
                st.success("✅ 학습자료 PDF 텍스트 저장 완료!")
                st.markdown("**📖 문서 미리보기 (일부)**")
                st.write(pdf_text[:500] + ("..." if len(pdf_text) > 500 else ""))
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("📚 저장된 학습자료 PDF 목록")
    try:
        pdf_list = supabase.table("pdf_texts").select("id, filename, created_at").order("created_at", desc=True).execute().data or []
    except Exception as e:
        st.error(f"저장된 학습자료 PDF 목록 불러오기 오류: {e}")
        pdf_list = []
    if pdf_list:
        for pdf in pdf_list:
            filename = pdf["filename"]
            created_raw = pdf.get("created_at")
            if created_raw:
                try:
                    dt_obj = datetime.datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                    kst = pytz.timezone("Asia/Seoul")
                    dt_kst = dt_obj.astimezone(kst)
                    created_formatted = dt_kst.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    created_formatted = created_raw
            else:
                created_formatted = "미상"
            st.markdown(f"""
            <div class="card" title="파일명: {filename}">
                <b>{filename}</b><br>
                업로드일: {created_formatted}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("아직 저장된 학습자료 PDF가 없습니다.")

elif selected_menu == "❓ 질의응답":
    st.subheader("❓ 질의응답")

    if api_choice == "Gemini":
        try:
            pdf_list = supabase.table("pdf_texts").select("id, filename").order("created_at", desc=True).execute().data or []
        except Exception as e:
            st.error(f"학습자료 PDF 목록 조회 오류: {e}")
            pdf_list = []

        if not pdf_list:
            st.info("먼저 '📄 학습자료 PDF 업로드' 메뉴에서 학습자료 PDF를 업로드하세요.")
        else:
            pdf_choice = st.selectbox("문서 선택", pdf_list, format_func=lambda x: x["filename"], key="qa_pdf_choice")
            if "user_question" not in st.session_state:
                st.session_state.user_question = ""
            st.session_state.user_question = st.text_input(
                "학습자료 기반 질문을 입력하세요:",
                value=st.session_state.user_question,
                key="user_question_input",
                placeholder="예: 이 문서에서 중요한 개념은 무엇인가요?"
            )
            if st.button("질문 제출", key="submit_question_gemini"):
                st.session_state.question_submitted = True

            if st.session_state.get("question_submitted", False):
                if st.session_state.user_question and pdf_choice:
                    with st.spinner("답변 생성 중..."):
                        answer = ask_gemini_with_pdf(st.session_state.user_question, gemini_model, pdf_choice["filename"])
                        st.info(f"💬 답변: {answer}")
                    st.session_state.question_submitted = False

    else:
        if "user_question_simple" not in st.session_state:
            st.session_state.user_question_simple = ""
        st.session_state.user_question_simple = st.text_input(
            "일반적인 질문을 입력하세요:",
            value=st.session_state.user_question_simple,
            key="user_question_simple_input",
            placeholder="예: AI란 무엇인가요?"
        )
        if st.button("질문 제출(OpenAI)", key="submit_question_openai"):
            st.session_state.question_submitted_simple = True

        if st.session_state.get("question_submitted_simple", False):
            if st.session_state.user_question_simple:
                with st.spinner("OpenAI 답변 생성 중..."):
                    answer = ask_openai_simple(st.session_state.user_question_simple, openai_client)
                    st.info(f"💬 답변: {answer}")
                st.session_state.question_submitted_simple = False

elif selected_menu == "📝 시험문제 출제":
    st.subheader("📝 시험문제 출제")
    try:
        pdf_list = supabase.table("pdf_texts").select("id, filename").order("created_at", desc=True).execute().data or []
    except Exception as e:
        st.error(f"학습자료 PDF 목록 조회 오류: {e}")
        pdf_list = []

    if not pdf_list:
        st.info("먼저 '📄 학습자료 PDF 업로드' 메뉴에서 학습자료 PDF를 업로드하세요.")
    else:
        pdf_choice = st.selectbox("출제할 문서 선택", pdf_list, format_func=lambda x: x["filename"], key="quiz_pdf_choice")
        if "unit_input_text" not in st.session_state:
            st.session_state.unit_input_text = ""
        st.session_state.unit_input_text = st.text_input(
            "단원명 입력 (선택사항)",
            value=st.session_state.unit_input_text,
            key="unit_input_key",
            placeholder="예: 단원1_기초이론"
        )
        if st.button("객관식 문제 생성하기", key="generate_quiz"):
            if pdf_choice:
                try:
                    selected_pdf_content = supabase.table("pdf_texts").select("content").eq("id", pdf_choice["id"]).single().execute().data["content"]
                except Exception as e:
                    st.error(f"학습자료 PDF 내용 조회 오류: {e}")
                    selected_pdf_content = ""

                if selected_pdf_content:
                    with st.spinner("퀴즈 생성 중..."):
                        quiz_text = generate_quiz_with_summary(selected_pdf_content, gemini_model, api_choice)
                        st.session_state.generated_quiz_raw = quiz_text
                        success = parse_and_save_quiz(quiz_text, st.session_state.unit_input_text.strip())
                        if not success:
                            st.error("퀴즈 저장에 실패했습니다.")
                else:
                    st.info("문서를 먼저 선택하세요.")
            else:
                st.warning("출제할 문서를 선택해 주세요.")

        if st.session_state.get("generated_quiz_raw"):
            st.markdown("### 📝 생성된 문제 미리보기")
            quiz_lines = [line.strip() for line in st.session_state["generated_quiz_raw"].splitlines() if line.strip()]
            q, opts, ans, expl = "", [], "", ""
            for line in quiz_lines:
                if line.startswith("문제:"):
                    q = line.replace("문제:", "").strip()
                elif line.startswith(("1.", "2.", "3.", "4.")):
                    opts.append(line[2:].strip())
                elif line.startswith("정답:"):
                    ans = line.replace("정답:", "").strip()
                elif line.startswith("해설:"):
                    expl = line.replace("해설:", "").strip()
            if q:
                st.write(f"**Q. {q}**")
            for idx, o in enumerate(opts, 1):
                st.write(f"{idx}. {o}")
            if ans:
                st.markdown(f"**정답:** {ans}")
            if expl:
                st.markdown(f"**해설:** {expl}")

elif selected_menu == "📑 문제 보기":
    st.subheader("📑 저장된 문제 보기")

    quizzes = get_quizzes() or []
    if not quizzes:
        st.info("저장된 문제가 없습니다.")
    else:
        items_per_page = 10
        total_items = len(quizzes)
        total_pages = (total_items - 1) // items_per_page + 1

        if "quiz_page" not in st.session_state:
            st.session_state["quiz_page"] = 1
        current_page = st.session_state["quiz_page"]

        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        current_quizzes = quizzes[start_idx:end_idx]

        for q in current_quizzes:
            unit = (q.get("unit") or "").strip()
            unit_display = f"[{unit}] " if unit else ""
            question_text = q.get("question", "")
            st.markdown(f"**{unit_display}Q. {question_text}**")

            try:
                opts = json.loads(q["options"]) if isinstance(q["options"], str) else q["options"]
            except Exception:
                opts = []

            correct_answer_raw = (q.get("answer") or "").strip()
            correct_answer = correct_answer_raw.lower()
            answer_idx = None

            for idx, opt in enumerate(opts, start=1):
                opt_stripped = opt.strip().lower()
                option_label = f"{idx}. {opt}"
                if correct_answer == str(idx) or correct_answer == opt_stripped:
                    st.markdown(f"- **{option_label}** ✅")
                    answer_idx = idx
                else:
                    st.markdown(f"- {option_label}")

            explanation = (q.get("explanation") or "").strip()
            if explanation and answer_idx is not None:
                st.markdown(f"**정답: {answer_idx}**  <br>  **해설:** {explanation}", unsafe_allow_html=True)
            elif explanation:
                st.markdown(f"**해설:** {explanation}")
            st.markdown("---")
            st.markdown("<br>", unsafe_allow_html=True)

        def render_styled_pagination(current_page, total_pages, key_prefix="pagination"):
            max_buttons = 10
            start_page = max(1, current_page - 4)
            end_page = min(total_pages, start_page + max_buttons - 1)
            start_page = max(1, end_page - max_buttons + 1)
            btn_count = end_page - start_page + 1

            cols = st.columns(btn_count, gap="small")
            page_clicked = None

            for i, page_num in enumerate(range(start_page, end_page + 1)):
                disabled = (page_num == current_page)
                with cols[i]:
                    if st.button(str(page_num), key=f"{key_prefix}_{page_num}", disabled=disabled):
                        page_clicked = page_num
            return page_clicked

        page_clicked = render_styled_pagination(current_page, total_pages, key_prefix="quiz_page")
        if page_clicked is not None and page_clicked != current_page:
            st.session_state["quiz_page"] = page_clicked
