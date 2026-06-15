"""LO3~6 - 도구 쓰는 Agent 챗봇 (대표 앱 2).

03_tool_calling/·04_custom_tool/ 챕터에서 만든 @tool 도구와
06_langgraph_agent/ 챕터에서 익힌 create_agent(ReAct Agent)를 챗 UI로 옮긴 예제입니다.

다루는 것:
  - @tool로 만든 도구 몇 개를 create_agent에 붙여 멀티턴 대화 (03_tool_calling/·04_custom_tool/에 대응)
  - create_agent 한 줄로 추론-행동-관찰(ReAct) 루프를 자동 수행 (06_langgraph_agent/03_create_agent.py에 대응)
  - 도구 호출 가시화: 모델이 어떤 도구를 어떤 인자로 불렀고(tool_calls),
    도구가 무엇을 돌려줬는지(ToolMessage)를 채팅 흐름에 펼쳐 보여줌 (이 앱의 핵심 학습 포인트)
  - 사이드바에서 시스템 프롬프트 편집·도구 on/off 토글·대화 초기화

실행법(로컬, uv):
  1) 의존성 설치:  uv sync
  2) 키 설정:      cp .env.example .env  후 .env에 OPENAI_API_KEY 입력
  3) 실행:         uv run streamlit run streamlit/02_tool_agent.py

키가 없으면 안내만 표시하고 멈춥니다 (코드에 키를 직접 적지 않습니다).
단기 메모리(대화를 모델이 스스로 요약·관리)는 다음 앱에서 다루므로,
여기서는 st.session_state에 대화 기록을 직접 쌓는 것으로 충분합니다.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent  # v1.0 권장 경로(한 줄 ReAct Agent)
# v1 권장 경로. langchain_core.messages에서 가져와도 동일하게 동작합니다.
from langchain.messages import HumanMessage, AIMessage, ToolMessage
# @tool은 함수를 도구로, ToolException은 도구 안에서 의도적으로 오류를 던질 때 씁니다.
from langchain_core.tools import tool, ToolException

# .env 파일이 있으면 환경변수로 읽어 들입니다 (로컬 실행 시 키를 .env에 둡니다).
load_dotenv()


# 모델은 "벤더:모델명" 문자열 하나로 지정합니다 (챕터 예제와 동일 규칙).
# google-genai로 전환하려면: MODEL = "google-genai:gemini-3.5-flash" (GOOGLE_API_KEY 필요)
MODEL = "openai:gpt-5.4-mini"


# ---------------------------------------------------------------------------
# 도구 정의 — 03_tool_calling/·04_custom_tool/에서 만든 @tool을 그대로 가져옵니다.
# docstring이 곧 도구 설명이며, 모델은 이 설명을 보고 어떤 도구를 부를지 결정합니다(라우팅 기준).
# ---------------------------------------------------------------------------

@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다."""
    return a * b


# 사내 재고를 흉내 낸 데모 데이터입니다 (실제로는 DB·API를 호출하는 자리입니다).
_STOCK = {("BAT-21700", "ICN"): 1240, ("BAT-21700", "PUS"): 380}


@tool
def check_inventory(sku: str, warehouse: str = "ICN") -> str:
    """지정한 창고의 제품 재고 수량을 조회한다.
    제품 코드(sku)와 창고 코드(warehouse)로 현재 보유 수량을 반환한다.
    예: 'BAT-21700 인천(ICN) 창고 재고'."""
    qty = _STOCK.get((sku.strip().upper(), warehouse.strip().upper()))
    if qty is None:
        # 데이터가 없으면 ToolException으로 회신해, 모델이 지어내지 않고 사용자에게 되묻게 합니다.
        raise ToolException(f"재고 정보 없음: sku={sku}, warehouse={warehouse}")
    return f"{warehouse} 창고의 {sku} 재고는 {qty}개입니다."


# 데모용 고정 날씨 데이터입니다 (실제로는 외부 날씨 API를 부르는 자리입니다).
_WEATHER = {"서울": "맑음, 22도", "도쿄": "흐림, 19도", "보스턴": "비, 15도"}


@tool
def get_weather(city: str) -> str:
    """주어진 도시의 현재 날씨를 한 줄로 알려준다. 예: '서울 날씨 어때?'."""
    return _WEATHER.get(city.strip(), f"{city}의 날씨 정보가 없습니다.")


# 도구 이름 -> (도구 객체, 한 줄 설명). 사이드바 토글과 안내 문구를 한곳에서 관리합니다.
TOOL_CATALOG = {
    "add": (add, "두 정수 덧셈"),
    "multiply": (multiply, "두 정수 곱셈"),
    "check_inventory": (check_inventory, "사내 재고 조회 (데모: BAT-21700 / ICN·PUS)"),
    "get_weather": (get_weather, "도시 날씨 조회 (데모: 서울·도쿄·보스턴)"),
}


st.set_page_config(page_title="도구 쓰는 Agent", page_icon="🛠️")
st.title("🛠️ 도구 쓰는 Agent")
st.caption(f"모델: {MODEL} · LO3~6 도구 호출과 ReAct Agent 실습")


# ---------------------------------------------------------------------------
# 사이드바: 시스템 프롬프트 편집 · 도구 on/off · 대화 초기화
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    system_prompt = st.text_area(
        "시스템 프롬프트",
        value=(
            "너는 도구를 활용하는 한국어 업무 비서다. "
            "계산·재고·날씨처럼 도구로 확인할 수 있는 값은 추측하지 말고 반드시 도구로 확인하라. "
            "도구가 실패하면 지어내지 말고 사용자에게 다시 확인을 요청하라. "
            "답변은 간결한 한국어 한두 문장으로 정리하라."
        ),
        height=160,
        help="Agent의 역할·규칙을 정합니다. 매 호출마다 대화 맨 앞에 전달됩니다.",
    )

    st.subheader("사용할 도구")
    st.caption("켜진 도구만 Agent에 연결됩니다. 도구를 끄면 모델은 그 기능을 쓸 수 없습니다.")
    enabled_tools = []
    for name, (tool_obj, desc) in TOOL_CATALOG.items():
        if st.checkbox(f"{name} — {desc}", value=True, key=f"tool_{name}"):
            enabled_tools.append(tool_obj)

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.pop("messages", None)
        st.rerun()


# 키가 없으면 안내만 표시하고 멈춥니다 (src와 동일한 안전 처리).
if not os.environ.get("OPENAI_API_KEY"):
    st.warning('OPENAI_API_KEY 환경변수가 필요합니다. .env에 키를 입력하거나 `export OPENAI_API_KEY="sk-..."` 후 다시 실행하십시오.')
    st.stop()


# 대화 기록은 LangChain 메시지 객체 리스트로 보관합니다.
# (역할, 텍스트) 튜플 대신 메시지 객체를 그대로 쌓아, tool_calls·ToolMessage 같은
# 도구 호출 흔적이 화면을 다시 그릴 때도 남아 있게 합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []  # [HumanMessage | AIMessage | ToolMessage, ...]


def render_tool_steps(steps):
    """한 턴에서 모델이 거친 도구 호출·결과(중간 단계)를 펼쳐 보여줍니다.

    steps는 (AIMessage 또는 ToolMessage) 리스트입니다.
      - tool_calls가 담긴 AIMessage  -> "도구 호출: 이름(인자)" 로 표시
      - ToolMessage                  -> 그 호출의 반환값(관찰)으로 표시
    이 가시화가 이 앱의 핵심 학습 포인트입니다.
    """
    # 표시할 호출이 하나도 없으면(도구 없이 바로 답한 경우) 굳이 빈 칸을 만들지 않습니다.
    has_calls = any(getattr(m, "tool_calls", None) for m in steps if isinstance(m, AIMessage))
    if not has_calls:
        return

    # 도구 결과(ToolMessage)를 호출 id로 찾기 위한 사전을 만듭니다.
    results_by_id = {m.tool_call_id: m.content for m in steps if isinstance(m, ToolMessage)}

    # 전체 호출 수를 세어 expander 제목에 표시합니다.
    total_calls = sum(len(m.tool_calls) for m in steps if isinstance(m, AIMessage) and m.tool_calls)
    with st.expander(f"도구 호출 과정 ({total_calls}건)", expanded=True):
        for m in steps:
            if not (isinstance(m, AIMessage) and m.tool_calls):
                continue
            for call in m.tool_calls:
                # 인자를 "a=3, b=5" 형태의 읽기 쉬운 문자열로 만듭니다.
                args_text = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                result = results_by_id.get(call["id"], "(결과 없음)")
                st.markdown(f"- **도구 호출**: `{call['name']}({args_text})`")
                st.markdown(f"  - **결과**: {result}")


def render_message(msg):
    """세션에 쌓인 메시지 하나를 화면에 그립니다 (사람·assistant 텍스트만 말풍선으로)."""
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        # 최종 자연어 답변만 assistant 말풍선으로 보여줍니다.
        # (도구 호출만 담긴 중간 AIMessage는 content가 비어 있어 여기서 걸러집니다.)
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# 지난 대화를 화면에 다시 그립니다.
# 사용자 입력 직후 이어지는 도구 호출 묶음을 모아 expander로 함께 보여 줍니다.
prev_len = len(st.session_state.messages)
i = 0
while i < prev_len:
    msg = st.session_state.messages[i]
    if isinstance(msg, HumanMessage):
        render_message(msg)
        # 이 사람 발화 다음부터 다음 사람 발화 전까지가 한 턴의 Agent 응답입니다.
        j = i + 1
        turn_steps = []
        while j < prev_len and not isinstance(st.session_state.messages[j], HumanMessage):
            turn_steps.append(st.session_state.messages[j])
            j += 1
        render_tool_steps(turn_steps)               # 도구 호출 과정 펼쳐 보기
        for step in turn_steps:
            render_message(step)                     # 최종 답변 말풍선
        i = j
    else:
        i += 1


# ---------------------------------------------------------------------------
# 사용자 입력 -> Agent 실행 -> 도구 호출 과정 + 최종 답변 표시
# ---------------------------------------------------------------------------
if user_input := st.chat_input("메시지를 입력하세요"):
    # 1) 사용자 메시지를 기록하고 즉시 화면에 표시합니다.
    st.session_state.messages.append(HumanMessage(user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) 이번 턴의 설정(시스템 프롬프트·켜진 도구)으로 Agent를 만듭니다.
    #    create_agent는 06_langgraph_agent/01_manual_agent_graph.py의 수동 그래프(모델 노드·ToolNode·조건부 엣지·순환)를
    #    내부에서 똑같이 구성합니다. 즉, ReAct 루프가 자동으로 돕니다.
    agent = create_agent(
        MODEL,
        tools=enabled_tools,         # 사이드바에서 켠 도구만 연결 (끄면 그 기능을 못 씀)
        system_prompt=system_prompt,  # 사이드바에서 편집한 역할·규칙
    )

    # 3) 지금까지의 대화 전체를 넘겨 멀티턴을 유지합니다.
    #    create_agent는 결과로 입력 메시지 + 이번 턴에 생성된 메시지를 모두 담아 돌려줍니다.
    before = len(st.session_state.messages)
    new_messages = []  # 호출 실패 시 st.stop()으로 멈추므로 기본값은 빈 리스트로 둡니다.
    with st.chat_message("assistant"):
        with st.spinner("Agent가 도구를 사용해 처리하고 있습니다..."):
            try:
                result = agent.invoke({"messages": st.session_state.messages})
                # 이번 턴에 새로 생긴 메시지(도구 호출·결과·최종 답변)만 골라 둡니다.
                # 입력으로 넘긴 메시지는 그대로 앞부분에 다시 담겨 오므로, before 이후만 취합니다.
                new_messages = result["messages"][before:]
            except Exception as e:  # 네트워크·키·한도 등 호출 실패를 사용자에게 그대로 안내
                err = f"Agent 실행 중 오류가 발생했습니다: {e}"
                st.error(err)
                st.session_state.messages.append(AIMessage(err))
                st.stop()

    # 4) 새로 생긴 메시지를 세션에 반영합니다.
    st.session_state.messages.extend(new_messages)

    # 5) 방금 턴의 도구 호출 과정을 펼쳐 보이고, 최종 답변을 말풍선으로 출력합니다.
    render_tool_steps(new_messages)
    for step in new_messages:
        render_message(step)
