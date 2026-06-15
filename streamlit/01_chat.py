"""LO2 - Streamlit 챗봇 (대표 앱 1).

02_langchain_core/ 챕터에서 익힌 LangChain 로직(메시지 구성, 스트리밍, 구조화 출력)을
간단한 챗 UI로 옮긴 예제입니다.

다루는 것:
  - st.chat_input / st.chat_message로 멀티턴 대화 (02_langchain_core/02_messages_context.py에 대응)
  - 사이드바에서 시스템 프롬프트·temperature 조절 (02_langchain_core/03_params_streaming.py에 대응)
  - st.write_stream으로 토큰 단위 스트리밍 출력 (02_langchain_core/03_params_streaming.py에 대응)
  - "구조화 출력 모드" 토글: 켜면 Pydantic 스키마로 인물 정보를 추출해 표로 표시
    (02_langchain_core/05_structured_output.py·06_structured_advanced.py에 대응)

실행법(로컬):
  1) uv sync                              # streamlit 포함 의존성 설치
  2) cp .env.example .env  후 OPENAI_API_KEY 입력 (코드에 키를 적지 않음)
  3) uv run streamlit run streamlit/01_chat.py
"""

import os
from typing import Optional, List

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field


# 모델은 "벤더:모델명" 문자열 하나로 지정합니다 (챕터 예제와 동일 규칙).
# google-genai로 전환하려면: MODEL = "google-genai:gemini-3.5-flash" (GOOGLE_API_KEY 필요)
MODEL = "openai:gpt-5.4-mini"


# 구조화 출력 모드에서 사용할 스키마 (02_langchain_core/06_structured_advanced.py의 중첩 스키마와 같은 구조)
class PersonInfo(BaseModel):
    name: str = Field(description="사람의 이름")
    age: Optional[int] = Field(default=None, description="만 나이, 모르면 비워 둔다")
    skills: List[str] = Field(default_factory=list, description="보유 기술 목록")
    company: Optional[str] = Field(default=None, description="소속 회사, 모르면 비워 둔다")


st.set_page_config(page_title="LangChain 챗봇", page_icon="💬")
st.title("💬 LangChain 챗봇")
st.caption(f"모델: {MODEL} · LO2 핵심 구성요소 실습")


# --- 사이드바: 호출 파라미터 조절 (02_langchain_core/03_params_streaming.py) ---
with st.sidebar:
    st.header("설정")
    system_prompt = st.text_area(
        "시스템 프롬프트",
        value="너는 친절한 한국어 비서다. 간결하게 답한다.",
        help="모델의 역할·규칙을 정합니다. 대화 맨 앞에 한 번만 전달됩니다.",
    )
    temperature = st.slider(
        "temperature",
        min_value=0.0, max_value=2.0, value=0.7, step=0.1,
        help="낮으면 일관된 답, 높으면 다양·창의적인 답을 냅니다.",
    )
    structured_mode = st.toggle(
        "구조화 출력 모드",
        value=False,
        help="켜면 입력에서 인물 정보(이름·나이·기술·회사)를 추출해 표로 보여줍니다.",
    )
    if st.button("대화 초기화"):
        st.session_state.pop("messages", None)
        st.rerun()


# 키가 없으면 안내만 표시하고 멈춥니다 (챕터 예제와 동일한 안전 처리).
if not os.environ.get("OPENAI_API_KEY"):
    st.warning('OPENAI_API_KEY 환경변수가 필요합니다. `export OPENAI_API_KEY="sk-..."` 후 다시 실행하십시오.')
    st.stop()


# 대화 기록은 세션 상태에 (역할, 텍스트) 튜플로 보관합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []  # [("user"|"assistant", "텍스트"), ...]


def build_messages():
    """세션의 대화 기록을 LangChain 메시지 리스트로 변환합니다 (02_langchain_core/02_messages_context.py와 동일 구조)."""
    msgs = [SystemMessage(system_prompt)]  # 시스템 프롬프트는 항상 맨 앞에 한 번
    for role, text in st.session_state.messages:
        msgs.append(HumanMessage(text) if role == "user" else AIMessage(text))
    return msgs


# 지난 대화를 화면에 다시 그립니다.
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)


# 사용자 입력을 받습니다.
if user_input := st.chat_input("메시지를 입력하세요"):
    # 사용자 메시지를 기록하고 화면에 표시
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    if structured_mode:
        # --- 구조화 출력 모드: Pydantic 스키마로 인물 정보 추출 → 표 (02_langchain_core/05_structured_output.py·06_structured_advanced.py) ---
        model = init_chat_model(MODEL, temperature=temperature)
        structured_model = model.with_structured_output(PersonInfo)
        with st.chat_message("assistant"):
            try:
                info = structured_model.invoke(user_input)
                # Pydantic 객체를 dict로 바꿔 표 한 줄로 보여줍니다.
                st.table({
                    "이름": [info.name],
                    "나이": [info.age if info.age is not None else "(없음)"],
                    "기술": [", ".join(info.skills) if info.skills else "(없음)"],
                    "회사": [info.company or "(없음)"],
                })
                summary = f"추출 결과: {info.name} / 나이 {info.age} / 기술 {info.skills} / 회사 {info.company}"
                st.session_state.messages.append(("assistant", summary))
            except Exception as e:
                # 입력에 인물 정보가 없으면 검증이 실패할 수 있습니다 (02_langchain_core/06_structured_advanced.py의 예외 처리).
                err = f"구조화에 실패했습니다. 인물 정보가 담긴 문장을 입력해 주세요. ({e})"
                st.error(err)
                st.session_state.messages.append(("assistant", err))
    else:
        # --- 일반 챗 모드: 멀티턴 + 스트리밍 (02_langchain_core/02_messages_context.py·03_params_streaming.py) ---
        model = init_chat_model(MODEL, temperature=temperature)
        with st.chat_message("assistant"):
            # st.write_stream은 제너레이터의 조각을 받아 화면에 흘려 출력하고,
            # 다 끝나면 합쳐진 전체 텍스트를 반환합니다.
            # chunk.text는 청크의 텍스트만 안전하게 꺼냅니다 (02 챕터 예제와 동일).
            stream = (chunk.text for chunk in model.stream(build_messages()))
            full_text = st.write_stream(stream)
        st.session_state.messages.append(("assistant", full_text))
