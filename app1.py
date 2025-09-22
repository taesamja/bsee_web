# =============================================================
# Streamlit App - EASY AI 전자회로 맞춤 학습 도우미
# Date: 2025-09-21 (last updated)
# =============================================================

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
from functools import wraps
import re
import time
from typing import Optional

# ----------------------------- 기본 설정 -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

ST_KST = pytz.timezone("Asia/Seoul")
PT = pytz.timezone("America/Los_Angeles")
DEFAULT_PROJECT = "my-app"   # api_usage 프로젝트 구분자

# ----------------------------- 페이지 설정 -----------------------------
st.set_page_config(page_title="PDF 기반 Q&A 시스템", layout="wide")




# ----------------------------- 스타일 (라이트 + PCB 테마) -----------------------------
st.markdown("""
<style>
:root{
  --bg:#f6f7fb; --text:#0f172a; --muted:#6b7280; --brand:#2563eb; --brand-2:#7c3aed;
  --card:#ffffffcc; --card-solid:#ffffff; --accent:#10b981; --warn:#f59e0b; --danger:#ef4444;
  --radius:16px; --shadow-sm:0 1px 2px rgba(0,0,0,.06); --shadow-md:0 6px 20px rgba(0,0,0,.08);
}

/* 라이트 고정 */
html, body, [class*="stAppViewContainer"], [class*="stApp"]{ background:var(--bg)!important; color:var(--text)!important; }

/* 사이드바 */
[data-testid="stSidebar"]{ background:var(--card-solid)!important; border-right:1px solid rgba(15,23,42,.06); }

/* 헤더 타이포 */
.big-font{
  font-size:2.25rem!important; font-weight:800; letter-spacing:-.02em;
  background:linear-gradient(120deg, var(--brand), var(--brand-2));
  -webkit-background-clip:text; background-clip:text; color:transparent!important; margin:0 0 .25rem 0;
  margin-top: 1rem !important;
}
.subcap{ color:var(--muted); font-size:.95rem; }

/* 카드/구분선 */
.card{ background:var(--card); backdrop-filter:blur(8px); border-radius:var(--radius);
  padding:1.0em 1.2em; box-shadow:var(--shadow-md); border:1px solid rgba(15,23,42,.06); color:var(--text); }
.divider{ border-top:1px solid rgba(15,23,42,.08); margin:1.4em 0; }

/* 답변 박스 */
.analysis-result{
  background:linear-gradient(180deg, rgba(16,185,129,.12), rgba(16,185,129,.08));
  padding:14px; border-radius:12px; border:1px solid rgba(16,185,129,.25); box-shadow:var(--shadow-sm);
}

/* 뱃지 */
.role-pill{ display:inline-flex; gap:.35rem; align-items:center; padding:4px 10px; border-radius:999px;
  background:#eaf2ff; color:#1d4ed8; font-weight:700; font-size:.9rem; }
.teacher-badge{ display:inline-flex; padding:4px 10px; border-radius:999px; background:#fff5e6; color:#b45309; font-weight:700; font-size:.85rem; }

/* 버튼 */
.stButton>button{
  border-radius:12px!important; padding:.6rem 1rem!important; border:1px solid rgba(15,23,42,.08)!important;
  background:linear-gradient(180deg, #fff, #f3f4f6)!important; color:var(--text)!important; box-shadow:var(--shadow-sm)!important;
  transition:transform .06s ease, box-shadow .2s ease!important;
}
.stButton>button:hover{ transform:translateY(-1px); box-shadow:var(--shadow-md)!important; }
.stButton>button:active{ transform:translateY(0); box-shadow:var(--shadow-sm)!important; }

/* 입력/선택 컴포넌트 */
.stRadio>div, .stSelectbox>div, .stTextInput>div, .stTextArea>div{
  border-radius:12px; border:1px solid rgba(15,23,42,.08); background:var(--card-solid); box-shadow:var(--shadow-sm);
}
.stTextInput input, .stTextArea textarea{ border-radius:10px!important; }

/* 보기 라디오(테두리/배경 제거 + 폰트 키움) */
.stRadio > div{ background:transparent!important; padding:0!important; box-shadow:none!important; border:none!important; }
.stRadio label{ font-size:1.1em!important; font-weight:500; }

/* 메트릭 */
[data-testid="stMetric"]{ background:var(--card); border:1px solid rgba(15,23,42,.06); border-radius:14px; padding:12px; box-shadow:var(--shadow-sm); }
[data-testid="stMetricValue"]{ color:var(--brand)!important; }

/* 데이터프레임 */
[data-testid="stDataFrame"]{ background:var(--card-solid)!important; border:1px solid rgba(15,23,42,.06)!important; border-radius:12px; box-shadow:var(--shadow-sm); overflow:hidden; }
[data-testid="stDataFrame"] [role="columnheader"]{ background:#f8fafc!important; font-weight:700; color:#0f172a!important; border-bottom:1px solid rgba(15,23,42,.06)!important; }
[data-testid="stDataFrame"] [role="row"]{ transition:background .15s ease; }
[data-testid="stDataFrame"] [role="row"]:hover{ background:#f9fafb!important; }

/* 탭/익스팬더 */
.stTabs [data-baseweb="tab-list"]{ border-bottom:1px solid rgba(15,23,42,.06); }
.streamlit-expanderHeader{ font-weight:700; }

/* 알림 (입체감 제거) */
.stAlert, .stAlert div, [data-testid="stAlert"], [data-testid="stAlert"]>div, [data-testid="stAlert"]>div[role="alert"]{
  border:none!important; box-shadow:none!important; background:#eaf2ff!important;
}

/* 페이지 여백 */
.block-container{ padding-top:1.0rem; }

/* 링크 버튼 */
a.btn-primary{ display:inline-flex; align-items:center; gap:.5rem; background:linear-gradient(120deg, var(--brand), var(--brand-2));
  color:#fff!important; text-decoration:none!important; padding:.6rem 1rem; border-radius:12px; box-shadow:var(--shadow-md); }
a.btn-primary:hover{ filter:brightness(1.05); }

/* === HERO (웹서비스용) =================================== */
.pcb-hero{
  position: relative;
  border: 1px solid rgba(15,23,42,.08);
  border-radius: 20px;
  padding: 22px 20px;              /* ← 내부 여백 ↑ */
  box-shadow: 0 10px 26px rgba(2,6,23,.06);
  overflow: hidden;                /* 데코가 넘치지 않게 */
  background:
    radial-gradient(1200px 400px at -10% -10%, rgba(124,58,237,.12), transparent 50%),
    radial-gradient(900px 300px at 110% 120%, rgba(37,99,235,.12), transparent 60%),
    linear-gradient(120deg, rgba(37,99,235,.08), rgba(16,185,129,.06));
}

/* 회로 트레이스(기존 after 유지) */
.pcb-hero::after{
  content:"";
  position:absolute; inset:0;
  background:
    repeating-linear-gradient(90deg, rgba(37,99,235,.08) 0 2px, transparent 2px 24px),
    repeating-linear-gradient(0deg, rgba(16,185,129,.08) 0 2px, transparent 2px 24px);
  mask-image: radial-gradient(1200px 400px at 30% 40%, #000 30%, transparent 70%);
  pointer-events:none;
}

/* 우상단 얇은 원형 데코(사용설명서 카드와 톤 맞춤) */
.pcb-hero::before{
  content:"";
  position:absolute; right:-28px; top:-28px;
  width: 150px; height:150px; border-radius:50%;
  border: 2px solid rgba(37,99,235,.16);
}

/* 제목 전용 여백/줄높이 */
.pcb-hero .hero-title{
  margin: 6px 0 8px 0;            /* ← 위아래 간격 */
  line-height: 1.15;
}

/* 보조 캡션과 간격 살짝 */
.pcb-hero .subcap{ margin-top: 2px; }

/* 칩 아이콘(기존과 동일) */
.chip{ width:44px; height:44px; display:inline-flex; align-items:center; justify-content:center;
  border-radius:10px; background:#1e293b; color:#a7f3d0; box-shadow:var(--shadow-sm);
  border:1px solid rgba(255,255,255,.08);
}
.chip svg{ width:26px; height:26px; }
.unit-badge{ display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .55rem;
  background:rgba(16,185,129,.12); color:#047857; border:1px solid rgba(16,185,129,.28); border-radius:999px; font-weight:700; font-size:.8rem; }

/* 문제 카드 */
.pcb-card{ border-radius:14px; padding:14px 16px; background:var(--card-solid); border:1px dashed rgba(37,99,235,.28);
  box-shadow:var(--shadow-sm); position:relative; overflow:hidden; }
.pcb-card::before{ content:""; position:absolute; right:-20px; top:-20px; width:120px; height:120px; border-radius:50%;
  border:2px solid rgba(37,99,235,.18); }
.pcb-card h4{ margin:0 0 8px 0; } .pcb-card .opts li{ margin:.2rem 0; }

/* 코드/수식 색상 */
.katex-display, .katex{ color:#0f172a!important; }
code, pre{ background:#0b1220!important; color:#e2e8f0!important; border-radius:10px!important; }
</style>
""", unsafe_allow_html=True)

# ---------------------- 공용 유틸 / 데코레이터 ----------------------
def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Error in {func.__name__}: {e}")
            st.error("작업 처리 중 오류가 발생했습니다. 다시 시도해주세요.")
            return None
    return wrapper

# --------------------------- 클라이언트 초기화 ---------------------------
@st.cache_resource
def init_supabase_client() -> Client:
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except KeyError as e:
        st.error(f"'{e.args[0]}' secrets.toml 파일에 설정해 주세요.")
        st.stop()

@st.cache_resource
def init_gemini_model():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel("gemini-2.5-flash-lite")
    except Exception:
        st.error("Gemini 모델 초기화 실패. API 키 및 연결을 확인해 주세요.")
        st.stop()

@st.cache_resource
def init_openai_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except KeyError as e:
        st.error(f"'{e.args[0]}' secrets.toml 파일에 설정해 주세요.")
        st.stop()

supabase = init_supabase_client()
gemini_model = init_gemini_model()
openai_client = init_openai_client()

# --------------------------- 인증 / 세션 ---------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {"is_authenticated": False, "username": "", "role": ""}

def set_auth(username: str, role: str):
    st.session_state.auth = {"is_authenticated": True, "username": username, "role": role}

def clear_auth():
    st.session_state.auth = {"is_authenticated": False, "username": "", "role": ""}
    st.query_params.clear()
    st.cache_data.clear()
    st.rerun()

def require_login(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.auth.get("is_authenticated"):
            st.error("로그인이 필요합니다.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def require_teacher(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.auth.get("is_authenticated") or st.session_state.auth.get("role") != "teacher":
            st.error("이 기능은 교사만 사용할 수 있습니다.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

# -------------------------- 날짜/쿼터 유틸 --------------------------
def _today_pacific_date():
    return datetime.datetime.now(tz=PT).date()

@st.cache_data(ttl=15)
def get_usage_count_today(service: str, model: str, project: str = DEFAULT_PROJECT) -> int:
    usage_date = _today_pacific_date().isoformat()
    qb = (supabase.table("api_usage").select("request_count")
          .eq("usage_date", usage_date).eq("service", service).eq("model", model).eq("project", project))
    try:
        if hasattr(qb, "maybe_single"):
            res = qb.maybe_single().execute()
            row = getattr(res, "data", None) or {}
            return int(row.get("request_count", 0) or 0)
        res = qb.limit(1).execute()
        data = getattr(res, "data", None) or []
        return int((data[0]["request_count"]) if data else 0)
    except Exception:
        logging.exception("get_usage_count_today() failed; returning 0")
        return 0

def _upsert_usage(service: str, model: str, project: str, inc: int = 1):
    usage_date = _today_pacific_date().isoformat()
    table = supabase.table("api_usage")
    table.upsert({
        "usage_date": usage_date, "service": service, "model": model, "project": project,
        "request_count": 0, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, on_conflict="usage_date,service,model,project").execute()

    qb = (table.select("request_count")
          .eq("usage_date", usage_date).eq("service", service).eq("model", model).eq("project", project))
    try:
        if hasattr(qb, "maybe_single"):
            res = qb.maybe_single().execute()
            row = getattr(res, "data", None) or {}
            current_val = int(row.get("request_count", 0) or 0)
        else:
            res = qb.limit(1).execute()
            data = getattr(res, "data", None) or []
            current_val = int((data[0]["request_count"]) if data else 0)
    except Exception:
        logging.exception("_upsert_usage: read current_val failed; assuming 0")
        current_val = 0

    table.update({
        "request_count": current_val + inc,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }).eq("usage_date", usage_date).eq("service", service).eq("model", model).eq("project", project).execute()

    st.cache_data.clear()

# ----------------- Gemini 에러 판별 & 폴백 유틸 -----------------
def _is_gemini_quota_error(e: Exception) -> bool:
    msg = str(e) if e else ""
    return ("Quota exceeded" in msg or "You exceeded your current quota" in msg
            or "generativelanguage.googleapis.com/generate_content_free_tier_requests" in msg
            or "ResourceExhausted" in msg or "429" in msg)

def _extract_retry_seconds(e: Exception) -> int:
    msg = str(e) if e else ""
    m = re.search(r"retry[_ ]delay\s*{\s*seconds:\s*(\d+)", msg)
    if m:
        return int(m.group(1))
    m2 = re.search(r"Please retry in\s*([0-9]+)", msg)
    if m2:
        return int(m2.group(1))
    return 60

def _gemini_blocked_until() -> Optional[float]:
    return st.session_state.get("gemini_block_until_ts")

def _set_gemini_block(seconds: int):
    st.session_state["gemini_block_until_ts"] = time.time() + max(30, seconds)

# -------- 안전 생성: Gemini 우선 + OpenAI 폴백 + 카운트 --------
def safe_generate_text_with_count(
    prompt: str, *, prefer: str = "Gemini",
    gemini_model_name: str = "gemini-2.5-flash-lite",
    openai_model_name: str = "gpt-4.1-nano",
    project: str = DEFAULT_PROJECT
) -> str:
    use_gemini = (prefer == "Gemini")
    blocked_until = _gemini_blocked_until()

    # Gemini
    if use_gemini and gemini_model and (not blocked_until or time.time() >= blocked_until):
        try:
            resp = gemini_model.generate_content(prompt)
            text = getattr(resp, "text", None)
            _upsert_usage("gemini", gemini_model_name, project, inc=1)
            if text:
                return text.strip()
            st.warning("Gemini 응답이 비어 있어 OpenAI로 전환합니다.")
        except Exception as e:
            if _is_gemini_quota_error(e):
                wait_s = _extract_retry_seconds(e)
                _set_gemini_block(wait_s)
                _upsert_usage("gemini", gemini_model_name, project, inc=1)
                st.warning(f"Gemini 쿼터 초과로 OpenAI로 자동 전환합니다. (약 {wait_s}초 후 재시도 가능)")
            else:
                st.warning(f"Gemini 오류로 OpenAI로 전환합니다: {e}")

    # OpenAI
    try:
        r = openai_client.chat.completions.create(
            model=openai_model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        _upsert_usage("openai", openai_model_name, project, inc=1)
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI 호출도 실패했습니다: {e}")
        return ""

# ------------------------ 비즈니스 로직 ------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

@st.cache_data
@handle_exceptions
def extract_text_from_pdf(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

def get_pdf_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@handle_exceptions
def check_pdf_exists(filename: str, content_hash: str) -> bool:
    teacher_username = st.session_state.auth["username"]
    res = (supabase.table("pdf_texts").select("id")
           .or_(f"filename.eq.{filename},content_hash.eq.{content_hash}")
           .eq("teacher_username", teacher_username).limit(1).execute())
    return bool(res.data)

@handle_exceptions
@require_teacher
def save_pdf_text(filename: str, content: str, content_hash: str):
    teacher_username = st.session_state.auth["username"]
    supabase.table("pdf_texts").insert({
        "filename": filename, "content": content, "content_hash": content_hash,
        "teacher_username": teacher_username, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }).execute()

def get_teacher_pdfs():
    teacher_username = st.session_state.auth["username"]
    try:
        return (supabase.table("pdf_texts").select("id, filename, created_at")
                .eq("teacher_username", teacher_username)
                .order("created_at", desc=True).execute().data or [])
    except Exception as e:
        st.error(f"PDF 목록 조회 오류: {e}")
        return []

def get_all_teacher_pdfs():
    try:
        return (supabase.table("pdf_texts").select("id, filename, teacher_username, created_at")
                .order("created_at", desc=True).execute().data or [])
    except Exception as e:
        st.error(f"PDF 목록 조회 오류: {e}")
        return []

@st.cache_data(ttl=1800)
@handle_exceptions
def ask_with_pdf(question: str, pdf_id: int, prefer: str = "Gemini") -> str:
    res = supabase.table("pdf_texts").select("content").eq("id", pdf_id).single().execute()
    if not res.data or not res.data.get("content"):
        return "관련 내용을 찾지 못했습니다."
    context = res.data["content"]
    prompt = f"""다음은 참고 문서 내용입니다:
{context}

위 내용을 참고하여 아래 질문에 답변해 주세요:

질문: {question}
"""
    return safe_generate_text_with_count(prompt, prefer=prefer)

@st.cache_data(ttl=1800)
@handle_exceptions
def summarize_text_with_ai(text_chunk: str, prefer: str = "Gemini") -> str:
    prompt = f"다음 텍스트를 간결하고 명확하게 요약해줘:\n\n{text_chunk}"
    return safe_generate_text_with_count(prompt, prefer=prefer)

# 퀴즈 파서 + 저장
@handle_exceptions
@require_teacher
def parse_and_save_quiz(quiz_text: str, _unit: str = "") -> bool:
    if not quiz_text or not isinstance(quiz_text, str) or not quiz_text.strip():
        st.error("❌ 생성된 문제 텍스트가 비어 있습니다.")
        return False
    lines = [line.strip() for line in quiz_text.splitlines() if line.strip()]
    question, options, answer, explanation = "", [], "", ""
    for line in lines:
        if line.startswith("문제:"):
            question = line.replace("문제:", "", 1).strip()
        elif re.match(r"^[1-4]\.\s*", line):
            options.append(re.sub(r"^[1-4]\.\s*", "", line).strip())
        elif line.startswith("정답:"):
            answer = line.replace("정답:", "", 1).strip()
        elif line.startswith("해설:"):
            explanation = line.replace("해설:", "", 1).strip()
    # 정답 동등/부분 매칭
    matched = None
    al = (answer or "").lower()
    for opt in options:
        if opt.lower() == al:
            matched = opt; break
    if matched is None:
        for opt in options:
            if al in opt.lower() or opt.lower() in al:
                matched = opt; break
    fixed_answer = matched if matched is not None else answer
    if question and len(options) == 4 and fixed_answer:
        save_quiz(question, options, fixed_answer, explanation, _unit)
        st.success("✅ 퀴즈가 성공적으로 저장되었습니다.")
        return True
    st.error("❌ 퀴즈 파싱 실패 또는 데이터 부족입니다.")
    st.code(quiz_text or "", language="text")
    return False

@handle_exceptions
@require_teacher
def save_quiz(question: str, options: list, answer: str, explanation: str = "", unit: str = ""):
    teacher_username = st.session_state.auth["username"]
    supabase.table("quiz_questions").insert({
        "question": question, "options": json.dumps(options, ensure_ascii=False),
        "answer": answer, "explanation": explanation, "unit": unit,
        "teacher_username": teacher_username, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }).execute()
    st.cache_data.clear()

@st.cache_data(ttl=600)
def get_quizzes(teacher_only: bool = False, viewer_username: str = "", viewer_role: str = ""):
    try:
        query = supabase.table("quiz_questions").select("*")
        if teacher_only and viewer_role == "teacher" and viewer_username:
            query = query.eq("teacher_username", viewer_username)
        response = query.order("created_at", desc=True).execute()
        return response.data or []
    except Exception as e:
        st.error(f"퀴즈 목록 조회 오류: {e}")
        return []

@require_teacher
def delete_quiz(quiz_id: int):
    try:
        teacher_username = st.session_state.auth["username"]
        check = supabase.table("quiz_questions").select("teacher_username").eq("id", quiz_id).single().execute()
        if check.data and check.data.get("teacher_username") == teacher_username:
            supabase.table("quiz_questions").delete().eq("id", quiz_id).execute()
            st.success("✅ 퀴즈가 삭제되었습니다.")
            st.cache_data.clear()
        else:
            st.error("❌ 다른 교사의 문제는 삭제할 수 없습니다.")
    except Exception as e:
        st.error(f"퀴즈 삭제 오류: {e}")

@require_teacher
def delete_pdf(pdf_id: int):
    try:
        teacher_username = st.session_state.auth["username"]
        check = supabase.table("pdf_texts").select("teacher_username").eq("id", pdf_id).single().execute()
        if check.data and check.data.get("teacher_username") == teacher_username:
            supabase.table("pdf_texts").delete().eq("id", pdf_id).execute()
            st.success("✅ PDF가 삭제되었습니다.")
            st.cache_data.clear()
        else:
            st.error("❌ 다른 교사의 자료는 삭제할 수 없습니다.")
    except Exception as e:
        st.error(f"PDF 삭제 오류: {e}")

# 결과/통계
@handle_exceptions
@require_login
def submit_quiz_result(username: str, question_id: int, selected: str, is_correct: bool):
    supabase.table("quiz_results").insert({
        "username": username, "question_id": question_id,
        "selected": selected, "is_correct": is_correct,
        "scored_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }).execute()
    st.cache_data.clear()

@st.cache_data(ttl=3)
@handle_exceptions
@require_login
def get_my_results(username: str):
    res = (supabase.table("quiz_results")
           .select("id, question_id, selected, is_correct, scored_at, quiz_questions(question, answer, unit)")
           .eq("username", username).order("scored_at", desc=True).execute())
    if hasattr(res, "data") and isinstance(res.data, list):
        return res.data
    r = (supabase.table("quiz_results").select("*")
         .eq("username", username).order("scored_at", desc=True).execute())
    return r.data or []

def get_teacher_statistics():
    try:
        pdf_stats = supabase.table("pdf_texts").select("teacher_username").execute().data or []
        quiz_stats = supabase.table("quiz_questions").select("teacher_username").execute().data or []
        pdf_by = {}
        for p in pdf_stats: pdf_by[p.get("teacher_username", "미상")] = pdf_by.get(p.get("teacher_username", "미상"), 0) + 1
        quiz_by = {}
        for q in quiz_stats: quiz_by[q.get("teacher_username", "미상")] = quiz_by.get(q.get("teacher_username", "미상"), 0) + 1
        return pdf_by, quiz_by
    except Exception as e:
        st.error(f"통계 조회 실패: {e}")
        return {}, {}

# ------------------------------ UI ------------------------------
# 히어로 헤더
st.markdown("""
<div class="pcb-hero">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:14px; flex-wrap:wrap;">
    <div style="display:flex; align-items:center; gap:14px;">
      <div class="chip" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none">
          <rect x="6" y="6" width="12" height="12" rx="2" stroke="currentColor" stroke-width="1.5"/>
          <path d="M3 8h3M3 12h3M3 16h3M18 8h3M18 12h3M18 16h3M8 3v3M12 3v3M16 3v3M8 18v3M12 18v3M16 18v3"
                stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </div>
      <div>
        <div class="big-font">EASY AI 전자회로 맞춤 학습 도우미</div>
        <div class="subcap">회로 이론 · 문제 풀이 · PDF 기반 Q&amp;A</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# 랜딩 사용자 가이드
def render_user_guide():
    st.markdown("""
    <div class="pcb-card" style="padding:18px 20px; margin-top:8px;">
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <div class="chip">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <rect x="7" y="7" width="10" height="10" rx="2" stroke-width="2"/>
            <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M17 7l2-2M5 19l2-2" stroke-width="2"/>
          </svg>
        </div>
        <div>
          <div class="big-font" style="margin:0;">EASY AI 전자회로 도우미 사용설명서</div>
          <div class="subcap">처음 오셨나요? 아래 안내대로 1분 만에 시작해요.</div>
        </div>
      </div>

      <ol style="line-height:1.7; margin:4px 0 14px 22px;">
        <li><b>좌측 사이드바</b>에서 <b>로그인</b> 또는 <b>회원가입</b>을 합니다. (역할: <code>student</code> 또는 <code>teacher</code>)</li>
        <li><b>질의응답</b>:
          <ul style="margin-top:4px;">
            <li><b>Gemini (문서 기반)</b> – 교사가 올린 PDF를 선택 후 그 내용으로 질문하세요.</li>
            <li><b>OpenAI (일반 질문)</b> – 자유 형식 질문을 입력해 일반 지식을 묻습니다.</li>
          </ul>
        </li>
        <li><b>문제 풀기</b>(학생) – 문제를 선택하고 <b>번호</b>로 정답을 고르세요. 결과는 즉시 저장됩니다.</li>
        <li><b>내 결과</b>(학생) – 단원/키워드/정오 필터로 본인 성과를 확인·다운로드할 수 있어요.</li>
        <li><b>PDF 업로드·출제·관리</b>(교사) – PDF로부터 요약→문항 생성, 보관·삭제 및 결과 대시보드 제공.</li>
      </ol>

      <div class="divider"></div>

      <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px;">
        <div class="pcb-card" style="border-style:solid;">
          <h4>빠른 시작 (학생)</h4>
          <ul class="opts" style="margin-left:18px; line-height:1.7;">
            <li>회원가입 ▶ 역할 <b>student</b></li>
            <li><b>질의응답</b>에서 <b>Gemini</b> 선택 후 문서 골라 질문</li>
            <li><b>문제 풀기</b> ▶ 번호로 선택 ▶ 즉시 채점</li>
            <li><b>내 결과</b>에서 CSV 다운로드</li>
          </ul>
        </div>
        <div class="pcb-card" style="border-style:solid;">
          <h4>빠른 시작 (교사)</h4>
          <ul class="opts" style="margin-left:18px; line-height:1.7;">
            <li>회원가입 ▶ 역할 <b>teacher</b></li>
            <li><b>PDF 업로드</b> ▶ <b>시험문제 출제</b>로 자동 생성</li>
            <li><b>문제 관리</b>에서 검토·삭제</li>
            <li><b>결과 대시보드</b>로 정답률 확인</li>
          </ul>
        </div>
      </div>

      <details style="margin-top:14px;">
        <summary style="cursor:pointer; font-weight:700;">자주 묻는 질문(FAQ)</summary>
        <div style="padding-top:10px; line-height:1.75;">
          <b>Q.</b> 학생도 PDF를 올리나요?<br/>
          <b>A.</b> 아니요. PDF 업로드는 <b>교사</b>만 가능하며 학생은 모든 교사 PDF를 열람해 질문할 수 있습니다.
          <br/><br/>
          <b>Q.</b> 모델 전환은 어떻게 하나요?<br/>
          <b>A.</b> 질의응답 화면에서 <b>Gemini(문서 기반)</b> 또는 <b>OpenAI(일반 질문)</b> 라디오 버튼으로 전환합니다.
        </div>
      </details>
    </div>
    """, unsafe_allow_html=True)

# URL 파라미터 → 로그인 복원
qp = st.query_params
if not st.session_state.auth.get("is_authenticated"):
    auth_flag = qp.get("auth", "0")
    u = qp.get("u", "")
    r = qp.get("r", "")
    if auth_flag == "1" and u and r in ("teacher", "student"):
        set_auth(u, r)

# ----------------------------- 사이드바 -----------------------------
with st.sidebar:
    st.markdown("### 👤 사용자")
    if not st.session_state.auth["is_authenticated"]:
        tab_login, tab_signup = st.tabs(["로그인", "회원가입"])
        with tab_login:
            lg_user = st.text_input("아이디", key="lg_user")
            lg_pw = st.text_input("비밀번호", type="password", key="lg_pw")
            if st.button("로그인"):
                if not lg_user or not lg_pw:
                    st.warning("아이디/비밀번호를 입력하세요.")
                else:
                    res = (supabase.table("users")
                           .select("username, password_hash, role")
                           .eq("username", lg_user).single().execute())
                    user = res.data
                    if not user:
                        st.error("존재하지 않는 사용자입니다.")
                    elif hash_password(lg_pw) != user["password_hash"]:
                        st.error("비밀번호가 올바르지 않습니다.")
                    else:
                        set_auth(user["username"], user["role"])
                        st.success(f"로그인 성공! ({user['role']})")
                        st.query_params.update({"auth": "1", "u": user["username"], "r": user["role"]})
                        st.cache_data.clear()
                        st.rerun()

            ############### 테스트 계정 표시 부분 시작 ############################
            st.markdown("""
                <div style="
                    margin-top:14px;
                    padding:12px;
                    border-radius:12px;
                    border:1px dashed rgba(37,99,235,.35);
                    background:#f9fafb;
                    font-size:0.9rem;
                ">
                  <b>🔑 테스트용 계정</b><br>
                  <b>아이디/비밀번호 </b><br>
                  교사01: <code>teacher01 / 1111</code><br>
                  교사02: <code>teacher02 / 1111</code><br>

                  <hr style="border:0; border-bottom:1px border-top:1px solid rgba(37,99,235,.3); margin:8px 0;">
                  
                  학생01: <code>student01 / 1111</code><br>
                  학생02: <code>student02 / 1111</code><br>
                  학생03: <code>student03 / 1111</code><br>
                  학생04: <code>student04 / 1111</code><br>
                  학생05: <code>student05 / 1111</code> 
                </div>
                """, unsafe_allow_html=True)
            ############### 테스트 계정 표시 부분 끝 ############################
       
        with tab_signup:
            sg_user = st.text_input("아이디", key="sg_user")
            sg_pw = st.text_input("비밀번호", type="password", key="sg_pw")
            role = st.selectbox("역할", ["student", "teacher"], index=0)
            if st.button("회원가입"):
                if not sg_user or not sg_pw or role not in ("teacher", "student"):
                    st.warning("아이디/비밀번호/역할을 정확히 입력하세요.")
                else:
                    exist = supabase.table("users").select("id").eq("username", sg_user).limit(1).execute()
                    if exist.data:
                        st.error("이미 존재하는 사용자명입니다.")
                    else:
                        pw_hash = hash_password(sg_pw)
                        supabase.table("users").insert({"username": sg_user, "password_hash": pw_hash, "role": role}).execute()
                        set_auth(sg_user, role)
                        st.query_params.update({"auth": "1", "u": sg_user, "r": role})
                        st.success(f"회원가입 및 자동 로그인 완료! ({role})")
                        st.cache_data.clear()
                        st.rerun()
    else:
        st.markdown(f"**{st.session_state.auth['username']}** 님")
        st.markdown(f"역할: **{st.session_state.auth['role']}**")
        if st.button("로그아웃"):
            clear_auth()

# ----------------------------- 메뉴/랜딩 -----------------------------
if not st.session_state.auth["is_authenticated"]:
    st.info("좌측 사이드바에서 로그인 또는 회원가입을 진행해 주세요.")
    render_user_guide()
    st.stop()

role = st.session_state.auth["role"]
username = st.session_state.auth["username"]

if role == "teacher":
    menu = ["📄 학습자료 PDF 업로드", "❓ 질의응답", "📝 시험문제 출제", "📑 문제 관리", "📊 결과 대시보드", "🛠️ 시스템 관리"]
else:
    menu = ["❓ 질의응답", "📝 문제 풀기", "📋 내 결과"]

selected = st.sidebar.radio("메뉴 선택", menu, index=0)

# ---------------- 교사: PDF 업로드 ----------------
if role == "teacher" and selected == "📄 학습자료 PDF 업로드":
    st.subheader(f"📄 학습자료 PDF 업로드 - {username} 교사님")
    uploaded = st.file_uploader("PDF 업로드 (2MB 이하)", type="pdf")
    if uploaded:
        if uploaded.size > 2 * 1024 * 1024:
            st.error("⚠️ 파일 크기가 2MB를 초과했습니다.")
        else:
            with st.spinner("PDF 처리 중..."):
                text = extract_text_from_pdf(uploaded.getvalue())
                pdf_hash = get_pdf_content_hash(text)
                if check_pdf_exists(uploaded.name, pdf_hash):
                    st.error("❌ 동일한 파일이 이미 업로드되어 있습니다.")
                else:
                    save_pdf_text(uploaded.name, text, pdf_hash)
                    st.success("✅ PDF 텍스트 저장 완료!")
                    st.metric("추출된 텍스트 길이", f"{len(text):,}자")
                    if st.checkbox("텍스트 미리보기"):
                        st.text_area("미리보기", value=(text[:1000] + ("..." if len(text) > 1000 else "")),
                                     height=200, disabled=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("📚 내가 업로드한 PDF 목록")
    lst = get_teacher_pdfs()
    if lst:
        for pdf in lst:
            filename = pdf["filename"]
            created_raw = pdf.get("created_at")
            if created_raw:
                try:
                    dt_obj = datetime.datetime.fromisoformat(created_raw.replace("Z", "+00:00")).astimezone(ST_KST)
                    created_fmt = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    created_fmt = created_raw
            else:
                created_fmt = "미상"
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            with c1:
                st.markdown(f"<div class='card'><b>{filename}</b><br>업로드일: {created_fmt}</div>", unsafe_allow_html=True)
            with c2:
                if st.button("🗑️ 삭제", key=f"del_pdf_{pdf['id']}"):
                    delete_pdf(pdf["id"])
                    st.rerun()
    else:
        st.info("아직 업로드한 PDF가 없습니다.")

# ---------------- 질의응답 (공통) ----------------
if selected == "❓ 질의응답":
    st.subheader(f"❓ 질의응답 - {username}")
    model_choice = st.radio("우선 모델", ["Gemini (문서 기반)", "OpenAI (일반 질문)"], horizontal=True)

    if model_choice.startswith("Gemini"):
        if role == "teacher":
            view_mode = st.radio("PDF 보기 모드", ["내 PDF만", "모든 교사 PDF"], horizontal=True, index=0)
            pdf_list = get_teacher_pdfs() if view_mode == "내 PDF만" else get_all_teacher_pdfs()
        else:
            view_mode = "모든 교사 PDF"
            pdf_list = get_all_teacher_pdfs()

        if not pdf_list:
            st.info("PDF가 없습니다. (교사는 먼저 업로드하세요.)" if role != "teacher" else "아직 업로드한 PDF가 없습니다.")
        else:
            if view_mode == "모든 교사 PDF":
                pdf_choice = st.selectbox("문서 선택", pdf_list,
                    format_func=lambda x: f"{x['filename']} (출처: {x.get('teacher_username','미상')})")
            else:
                pdf_choice = st.selectbox("문서 선택", pdf_list, format_func=lambda x: x["filename"])

            file_label = pdf_choice.get("filename") if isinstance(pdf_choice, dict) else str(pdf_choice)
            st.markdown(f"<div>📄 <strong>{file_label}</strong></div>", unsafe_allow_html=True)

            q = st.text_input("질문 입력 (선택 문서 내용 기반으로 답변합니다)", placeholder="예: 중요 개념은?")
            if st.button("🚀 질문 제출 (Gemini)"):
                if not q.strip():
                    st.warning("질문을 입력하세요.")
                else:
                    with st.spinner("생성 중..."):
                        answer = ask_with_pdf(q, pdf_choice["id"], prefer="Gemini") or ""
                        if not answer:
                            st.error("답변 생성에 실패했습니다. 잠시 후 다시 시도해 주세요.")
                        else:
                            st.markdown(
                                f"<div class='analysis-result'><h4 style='margin:0 0 6px 0;'>🧩 회로 해설 (Gemini)</h4><div style='line-height:1.6'>{answer}</div></div>",
                                unsafe_allow_html=True
                            )
    else:
        q = st.text_area("일반 질문 입력", placeholder="예: RC 회로의 시간상수는 무엇인가요?", height=120)
        if st.button("🚀 질문 제출 (OpenAI)"):
            if not q.strip():
                st.warning("질문을 입력하세요.")
            else:
                with st.spinner("생성 중..."):
                    answer = safe_generate_text_with_count(q, prefer="OpenAI") or ""
                    if not answer:
                        st.error("답변 생성에 실패했습니다. 잠시 후 다시 시도해 주세요.")
                    else:
                        st.markdown(
                            f"<div class='analysis-result'><h4 style='margin:0 0 6px 0;'>🧩 회로 해설 (OpenAI)</h4><div style='line-height:1.6'>{answer}</div></div>",
                            unsafe_allow_html=True
                        )

# ---------------- 교사: 시험문제 출제 ----------------
if role == "teacher" and selected == "📝 시험문제 출제":
    st.subheader(f"📝 시험문제 출제 - {username} 교사님")
    lst = get_teacher_pdfs()
    if not lst:
        st.info("먼저 PDF를 업로드하세요.")
    else:
        pdf_with_content = []
        for p in lst:
            c = supabase.table("pdf_texts").select("content").eq("id", p["id"]).single().execute().data
            if c and c.get("content"):
                p["content"] = c["content"]; pdf_with_content.append(p)
        if not pdf_with_content:
            st.error("PDF 내용을 불러올 수 없습니다.")
        else:
            pdf_choice = st.selectbox("출제할 문서", pdf_with_content, format_func=lambda x: x["filename"])
            unit = st.text_input("단원명(선택)", placeholder="예: 단원 1")
            model_choice = st.radio("우선 모델", ["Gemini", "OpenAI"], horizontal=True)
            if st.button("문제 1문항 생성"):
                with st.spinner("생성 중..."):
                    quiz_text = summarize_text_with_ai(pdf_choice["content"], prefer=model_choice)  # 요약
                    # 요약을 토대로 문제 생성
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
                    {quiz_text}
                    """.strip()
                    quiz_full = safe_generate_text_with_count(prompt, prefer=model_choice)
                    if not quiz_full:
                        st.error("문제 생성에 실패했습니다. 모델을 변경하거나 다시 시도하세요.")
                    else:
                        st.code(quiz_full, language="text")
                        if parse_and_save_quiz(quiz_full, unit.strip()):
                            st.success("문제 저장 완료!")

# ---------------- 교사: 문제 관리 ----------------
if role == "teacher" and selected == "📑 문제 관리":
    st.subheader(f"📑 문제 관리 - {username} 교사님")
    view_mode = st.radio("보기 모드", ["내 문제만", "전체 문제"], horizontal=True)
    teacher_only = (view_mode == "내 문제만")
    quizzes = get_quizzes(teacher_only=teacher_only, viewer_username=username, viewer_role=role) or []
    if not quizzes:
        st.info("아직 출제한 문제가 없습니다." if teacher_only else "저장된 문제가 없습니다.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            unit_filter = st.selectbox("단원 필터", ["전체"] + sorted({(q.get("unit") or "").strip() for q in quizzes if (q.get("unit") or "").strip()}))
        with c2:
            search_term = st.text_input("문제 검색", placeholder="키워드")
        filtered = quizzes
        if unit_filter != "전체":
            filtered = [q for q in filtered if (q.get("unit") or "").strip() == unit_filter]
        if search_term:
            filtered = [q for q in filtered if search_term.lower() in q.get("question", "").lower()]
        for q in filtered:
            quiz_teacher = q.get("teacher_username", "미상")
            unit = (q.get("unit") or "").strip()
            if not teacher_only:
                st.markdown(f"**[{unit or '무단원'}] Q. {q.get('question','')}** <span class='teacher-badge'>출제: {quiz_teacher}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**[{unit or '무단원'}] Q. {q.get('question','')}**")
            try:
                opts = json.loads(q["options"]) if isinstance(q["options"], str) else q["options"]
            except Exception:
                opts = []
            ans = (q.get("answer") or "").strip().lower()
            for i, o in enumerate(opts, 1):
                label = f"{i}. {o}"
                if ans == o.strip().lower():
                    st.markdown(f"- **{label}** ✅")
                else:
                    st.markdown(f"- {label}")
            if q.get("explanation"):
                st.caption(f"해설: {q['explanation']}")
            if q.get("teacher_username") == username:
                _, cbtn = st.columns([6, 1])
                with cbtn:
                    if st.button("🗑️", key=f"del_q_{q['id']}"):
                        delete_quiz(q["id"]); st.rerun()
            st.markdown("---")

# ---------------- 교사: 결과 대시보드 ----------------
if role == "teacher" and selected == "📊 결과 대시보드":
    st.subheader(f"📊 결과 대시보드 - {username} 교사님")
    result_view = st.radio("결과 보기", ["모든 결과", "내 문제 결과만"], horizontal=True)
    try:
        if result_view == "내 문제 결과만":
            my_quizzes = supabase.table("quiz_questions").select("id").eq("teacher_username", username).execute().data or []
            my_ids = [q["id"] for q in my_quizzes]
            if my_ids:
                results = (supabase.table("quiz_results")
                    .select("id, username, selected, is_correct, scored_at, question_id, quiz_questions(question,answer,teacher_username)")
                    .in_("question_id", my_ids).order("scored_at", desc=True).execute().data or [])
            else:
                results = []
        else:
            results = (supabase.table("quiz_results")
                .select("id, username, selected, is_correct, scored_at, question_id, quiz_questions(question,answer,teacher_username)")
                .order("scored_at", desc=True).execute().data or [])
    except Exception as e:
        st.error(f"결과 조회 오류: {e}")
        results = []

    if not results:
        st.info("학생 풀이 결과가 없습니다.")
    else:
        df = pd.DataFrame(results)
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("총 응답 수", f"{len(df)}개")
        with col2:
            if "is_correct" in df.columns: st.metric("평균 정답률", f"{(df['is_correct'].mean() * 100.0):.1f}%")
        with col3:
            if "username" in df.columns: st.metric("참여 학생 수", f"{df['username'].nunique()}명")

        def _extract_question(row):
            qq = row.get("quiz_questions")
            return qq.get("question", "") if isinstance(qq, dict) else row.get("question") or ""

        def _extract_answer(row):
            qq = row.get("quiz_questions")
            return qq.get("answer", "") if isinstance(qq, dict) else row.get("answer") or ""

        def _fmt_bool_icon(v): return "✅ 정답" if v is True else ("❌ 오답" if v is False else "—")

        def _fmt_ts_kst(ts):
            try: return pd.to_datetime(ts, utc=True).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")
            except Exception: return ts

        pretty_df = pd.DataFrame({
            "학생": df.get("username", pd.Series([""]*len(df))),
            "문제": df.apply(_extract_question, axis=1).apply(lambda s: (s[:80]+"…") if isinstance(s,str) and len(s)>80 else s),
            "선택한 답": df.get("selected", pd.Series([""]*len(df))),
            "정답": df.apply(_extract_answer, axis=1),
            "채점": df.get("is_correct", pd.Series([None]*len(df))).apply(_fmt_bool_icon),
            "제출시각(KST)": df.get("scored_at", pd.Series([""]*len(df))).apply(_fmt_ts_kst),
        })

        st.markdown("#### 🧾 응답 상세")
        st.dataframe(pretty_df, use_container_width=True, hide_index=True)
        csv_bytes = pretty_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 보기용 CSV 다운로드", data=csv_bytes, file_name="quiz_results_pretty.csv", mime="text/csv")

        st.markdown("#### 👥 학생별 정답률 요약")
        if "username" in df.columns and "is_correct" in df.columns:
            grp = (df.groupby("username")["is_correct"].agg(시도수="count", 정답수="sum", 정답률=lambda s: round(100*s.mean(), 1))).reset_index().rename(columns={"username":"학생"})
            st.dataframe(grp, use_container_width=True, hide_index=True)

# ---------------- 교사: 시스템 관리 ----------------
if role == "teacher" and selected == "🛠️ 시스템 관리":
    st.subheader("🛠️ 시스템 관리 (교사 전용)")

    # --- 시스템 통계 ---
    st.markdown("### 📊 시스템 통계")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 내 자료")
        try:
            my_pdf = len(supabase.table("pdf_texts").select("id").eq("teacher_username", username).execute().data or [])
            my_quiz = len(supabase.table("quiz_questions").select("id").eq("teacher_username", username).execute().data or [])
            st.metric("내 PDF", f"{my_pdf}개")
            st.metric("내 퀴즈", f"{my_quiz}개")
        except Exception as e:
            st.error(f"통계 조회 실패: {e}")
    with c2:
        st.markdown("#### 전체 자료")
        try:
            total_pdf = len(supabase.table("pdf_texts").select("id").execute().data or [])
            total_quiz = len(supabase.table("quiz_questions").select("id").execute().data or [])
            st.metric("전체 PDF", f"{total_pdf}개")
            st.metric("전체 퀴즈", f"{total_quiz}개")
        except Exception as e:
            st.error(f"통계 조회 실패: {e}")

    # --- 교사별 기여도 ---
    st.markdown("### 👥 교사별 기여도")
    pdf_by, quiz_by = get_teacher_statistics()  # 이미 정의된 헬퍼 사용
    all_t = set(pdf_by.keys()) | set(quiz_by.keys())
    if all_t:
        import pandas as pd
        sdf = pd.DataFrame([{
            "교사": t,
            "PDF 수": pdf_by.get(t, 0),
            "퀴즈 수": quiz_by.get(t, 0),
            "총 기여": pdf_by.get(t, 0) + quiz_by.get(t, 0)
        } for t in all_t]).sort_values("총 기여", ascending=False)
        st.dataframe(sdf, use_container_width=True, hide_index=True)
    else:
        st.info("집계할 데이터가 아직 없습니다.")

    # --- 관리 도구 ---
    st.markdown("### 🔧 관리 도구")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🗑️ 데이터 정리")
        st.caption("※ 되돌릴 수 없습니다. 신중히 사용하세요.")
        if st.button("내 퀴즈 모두 삭제"):
            try:
                supabase.table("quiz_questions").delete().eq("teacher_username", username).execute()
                st.success("✅ 내 퀴즈가 모두 삭제되었습니다.")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"삭제 실패: {e}")

        if st.button("내 PDF 모두 삭제"):
            try:
                supabase.table("pdf_texts").delete().eq("teacher_username", username).execute()
                st.success("✅ 내 PDF가 모두 삭제되었습니다.")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"삭제 실패: {e}")

    with c2:
        st.markdown("#### 🔄 시스템 유지보수")
        if st.button("캐시 초기화"):
            st.cache_data.clear()
            st.success("✅ 캐시가 초기화되었습니다.")

        if st.button("API 연결 테스트"):
            # Gemini
            try:
                _ = gemini_model.generate_content("ping")
                st.success("✅ Gemini API 정상")
            except Exception as e:
                st.error(f"❌ Gemini API 오류: {e}")

            # OpenAI
            try:
                _ = openai_client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                )
                st.success("✅ OpenAI API 정상")
            except Exception as e:
                st.error(f"❌ OpenAI API 오류: {e}")

            # Supabase
            try:
                _ = supabase.table("pdf_texts").select("id").limit(1).execute()
                st.success("✅ Supabase 연결 정상")
            except Exception as e:
                st.error(f"❌ Supabase 연결 오류: {e}")


# ---------------- 학생: 문제 풀기 ----------------
if role == "student" and selected == "📝 문제 풀기":
    st.subheader("📝 문제 풀기 (학생)")
    all_q = get_quizzes(teacher_only=False, viewer_username=username, viewer_role=role) or []
    if not all_q:
        st.info("출제된 문제가 없습니다.")
    else:
        teachers = sorted({q.get("teacher_username", "미상") for q in all_q})
        c1, c2, c3 = st.columns(3)
        with c1: f_teacher = st.selectbox("출제 교사", ["전체"] + teachers)
        with c2: f_unit = st.selectbox("단원", ["전체"] + sorted({(q.get("unit") or '').strip() for q in all_q if (q.get("unit") or '').strip()}))
        with c3: f_search = st.text_input("문제 검색", placeholder="키워드")

        filtered = all_q
        if f_teacher != "전체": filtered = [q for q in filtered if q.get("teacher_username") == f_teacher]
        if f_unit != "전체":   filtered = [q for q in filtered if (q.get("unit") or "").strip() == f_unit]
        if f_search:          filtered = [q for q in filtered if f_search.lower() in q.get("question", "").lower()]

        if not filtered:
            st.info("조건에 맞는 문제가 없습니다.")
        else:
            q_obj = st.selectbox(
                "문제 선택", filtered,
                format_func=lambda x: f"[{(x.get('unit') or '무단원')}] {x.get('question','')[:40]}... (출제: {x.get('teacher_username','미상')})"
            )
            try:
                options = json.loads(q_obj["options"]) if isinstance(q_obj["options"], str) else q_obj["options"]
            except Exception:
                options = []

            st.markdown(f"**Q. {q_obj.get('question','')}**")
            st.caption(f"출제: {q_obj.get('teacher_username', '미상')} 교사")

            # 번호(①~) + 보기 라벨
            def _circled(n:int)->str: return chr(9311 + n)  # ①: 9312(=9311+1)
            option_labels = [f"{_circled(i)} {opt}" for i, opt in enumerate(options, 1)]
            value_map = {option_labels[i-1]: i for i in range(1, len(options)+1)}

            selected_label = st.radio("정답 선택", option_labels, index=None, label_visibility="collapsed")

            if st.button("제출"):
                if not selected_label:
                    st.warning("보기를 선택하세요.")
                else:
                    selected_no = value_map[selected_label]
                    selected_text = options[selected_no - 1]
                    answer_text = (q_obj.get("answer", "") or "").strip().lower()
                    correct = (answer_text == selected_text.strip().lower())
                    submit_quiz_result(username, q_obj["id"], selected_text, bool(correct))
                    if correct:
                        st.success(f"정답입니다! 🎉 (선택: {selected_no}번)")
                    else:
                        st.error(f"오답입니다. 정답: {q_obj.get('answer','')}")
                    if q_obj.get("explanation"):
                        st.info(f"해설: {q_obj['explanation']}")

# ---------------- 학생: 내 결과 ----------------
if role == "student" and selected == "📋 내 결과":
    st.subheader(f"📋 내 결과 - {username} 학생")
    if st.button("🔄 새로고침"): st.cache_data.clear(); st.rerun()

    rows = get_my_results(username) or []
    if not rows:
        st.info("풀이 기록이 없습니다.")
    else:
        df = pd.DataFrame(rows)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("총 풀이 문제", f"{len(df)}개")
        with c2:
            if "is_correct" in df.columns: st.metric("정답률", f"{(df['is_correct'].mean() * 100.0):.1f}%")
        with c3:
            if "is_correct" in df.columns: st.metric("맞춘 문제", f"{int(df['is_correct'].sum())}개")

        def _extract_question(row):
            qq = row.get("quiz_questions")
            return qq.get("question", "") if isinstance(qq, dict) else row.get("question") or ""

        def _extract_answer(row):
            qq = row.get("quiz_questions")
            return qq.get("answer", "") if isinstance(qq, dict) else row.get("answer") or ""

        def _extract_unit(row):
            qq = row.get("quiz_questions")
            return (qq.get("unit") or "").strip() if isinstance(qq, dict) else (row.get("unit") or "").strip()

        def _fmt_bool_icon(v): return "✅ 정답" if v is True else ("❌ 오답" if v is False else "—")

        def _fmt_ts_kst(ts):
            try: return pd.to_datetime(ts, utc=True).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")
            except Exception: return ts

        w = pd.DataFrame({
            "단원": df.apply(_extract_unit, axis=1),
            "문제": df.apply(_extract_question, axis=1),
            "내 선택": df.get("selected", pd.Series([""]*len(df))),
            "정답": df.apply(_extract_answer, axis=1),
            "정오표시": df.get("is_correct", pd.Series([None]*len(df))),
            "제출시각_raw": df.get("scored_at", pd.Series([""]*len(df))),
        })

        fc1, fc2, fc3 = st.columns([2, 2, 3])
        with fc1:
            units = ["전체"] + sorted({u for u in w["단원"].unique() if u})
            sel_unit = st.selectbox("단원 필터", units, index=0)
        with fc2:
            sel_result = st.radio("정답 보기", ["전체", "정답만", "오답만"], horizontal=True)
        with fc3:
            keyword = st.text_input("키워드 검색 (문제/정답/내 선택)")

        filt = w.copy()
        if sel_unit != "전체":  filt = filt[filt["단원"] == sel_unit]
        if sel_result == "정답만":  filt = filt[filt["정오표시"] == True]
        elif sel_result == "오답만": filt = filt[filt["정오표시"] == False]
        if keyword:
            kw = keyword.lower()
            filt = filt[
                filt["문제"].str.lower().str.contains(kw, na=False) |
                filt["정답"].str.lower().str.contains(kw, na=False) |
                filt["내 선택"].str.lower().str.contains(kw, na=False)
            ]

        pretty_df = pd.DataFrame({
            "단원": filt["단원"].fillna("").replace("", "무단원"),
            "문제": filt["문제"].apply(lambda s: (s[:80]+"…") if isinstance(s,str) and len(s)>80 else s),
            "내 선택": filt["내 선택"],
            "정답": filt["정답"],
            "채점": filt["정오표시"].apply(_fmt_bool_icon),
            "제출시각(KST)": filt["제출시각_raw"].apply(_fmt_ts_kst),
        })

        st.markdown("#### 🧾 최근 풀이 기록")
        st.dataframe(pretty_df, use_container_width=True, hide_index=True)

        csv_bytes = pretty_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 내 결과 CSV 다운로드", data=csv_bytes, file_name="my_quiz_results.csv", mime="text/csv")

        st.markdown("#### 🗂️ 최근 10개 결과 요약")
        st.dataframe(pretty_df.head(10), use_container_width=True, hide_index=True)
        st.caption("💡 최신 풀이가 상단에 표시되며, 시간은 KST 기준입니다.")
