"""LO7·LO8 - 기억하는 Agent 챗봇 (대표 앱 3).

07_short_memory/ 챕터(단기 메모리)와 08_long_memory/ 챕터(장기 메모리)에서 익힌
로직을 하나의 챗 UI로 옮긴 예제입니다. 한 화면에서 두 종류의 기억을 직접 체감합니다.

다루는 것:
  - 단기 메모리(InMemorySaver checkpointer)와 thread_id로 대화 맥락을 유지·분리
    (07_short_memory/02_checkpointer.py·03_thread_id.py에 대응)
  - 장기 메모리(InMemoryStore, 시맨틱 인덱스)를 In-graph 방식으로 회상해 시스템 프롬프트에 주입
    (08_long_memory/02_semantic_index.py·05_in_graph_recall.py·07_short_plus_long.py에 대응)
  - 사이드바에서 thread_id를 선택·입력 → 같은 thread면 맥락 유지, 다른 thread면 끊김을 체감
  - 사이드바에서 현재 장기 기억 목록을 확인하고, 자연어로 새 기억을 직접 추가(put)
  - 새 thread로 바꿔도 장기 기억은 회상됨을 확인 (단기는 thread별, 장기는 전체 공유)

실행법(로컬, uv):
  1) 의존성 설치:  uv sync
  2) 키 설정:      cp .env.example .env  후 .env에 OPENAI_API_KEY 입력
                   (임베딩과 모델 모두 OpenAI 키를 사용합니다)
  3) 실행:         uv run streamlit run streamlit/03_memory_agent.py

키가 없으면 안내만 표시하고 멈춥니다 (문법 점검은 키 없이도 가능).
InMemorySaver와 InMemoryStore는 프로세스 메모리에만 저장되므로, 앱을 종료하면 모든 기억이 사라집니다(데모용).
"""

import os
from typing import Annotated

from typing_extensions import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings  # v1 경로
from langchain.messages import SystemMessage  # v1 경로
from langgraph.checkpoint.memory import InMemorySaver  # 단기 메모리
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# .env 파일이 있으면 환경변수로 읽어 들입니다 (로컬 실행 시 키를 .env에 둡니다).
load_dotenv()


# 모델은 "벤더:모델명" 문자열 하나로 지정합니다 (챕터 예제와 동일 규칙).
# google-genai로 전환하려면: MODEL = "google-genai:gemini-3.5-flash" (GOOGLE_API_KEY 필요)
MODEL = "openai:gpt-5.4-mini"  # 강의 직전 최신 모델·가격을 재확인하십시오.

# 임베딩 모델. 시맨틱 인덱스가 텍스트를 벡터로 바꿀 때 씁니다 (dims는 이 모델에 맞춰 1536).
EMBED = "openai:text-embedding-3-small"

# 장기 기억을 담는 네임스페이스. (사용자, 주제) 두 칸으로 공간을 나눕니다. 첫 칸이 사용자 ID입니다.
NS = ("user-123", "memories")


# 그래프 상태: 단기 메모리는 메시지 누적으로 표현합니다 (08_long_memory/05_in_graph_recall.py의 State와 동일).
class State(TypedDict):
    messages: Annotated[list, add_messages]


def _text(value):
    """Store 값에서 텍스트를 안전하게 꺼냅니다.

    직접 put한 {"text": ...}와 다른 형태({"content": ...} 등)가 섞일 수 있으므로
    text → content → 통째로 문자열화 순으로 방어적으로 읽습니다 (08_long_memory/05_in_graph_recall.py의 _text와 동일).
    """
    if isinstance(value, dict):
        return value.get("text") or value.get("content") or str(value)
    return str(value)


@st.cache_resource
def build_agent():
    """단기(InMemorySaver) + 장기(InMemoryStore, 시맨틱 인덱스)를 장착한 Agent를 1회 생성합니다.

    @st.cache_resource로 캐시하므로 앱이 재실행(rerun)되어도 같은 checkpointer·store 객체가
    유지됩니다. 즉, 대화 맥락(단기)과 장기 기억이 세션 내내 누적됩니다.
    store 객체도 함께 반환해, 사이드바에서 목록 표시·수동 추가에 직접 사용합니다.
    """
    # 시맨틱 인덱스를 켜면 put할 때 지정한 필드(text)의 텍스트가 자동으로 벡터로 변환됩니다.
    store = InMemoryStore(
        index={
            "dims": 1536,  # text-embedding-3-small의 출력 차원 (embed와 한 쌍으로 맞춰야 함)
            "embed": init_embeddings(EMBED),  # 임베딩 모델 객체
            "fields": ["text"],  # 임베딩할 필드 지정 (value의 text만 벡터화)
        }
    )

    model = init_chat_model(MODEL)

    # In-graph 방식: 노드가 직접 store.search로 회상해 시스템 프롬프트에 끼워 넣습니다.
    # (08_long_memory/05_in_graph_recall.py와 동일한 패턴 — 회상 시점·개수를 코드가 100% 제어)
    def agent_node(state: State, *, store: BaseStore):
        last = state["messages"][-1].content
        # 마지막 발화와 의미가 가까운 장기 기억을 상위 3개만 회상합니다.
        recalled = store.search(NS, query=last, limit=3)
        memory_text = "\n".join(f"- {_text(r.value)}" for r in recalled) or "(없음)"
        sys = SystemMessage(
            "너는 사용자를 기억하는 비서다. 아래는 회상한 사실이며, "
            "질문과 어긋나면 무시하라.\n" + memory_text
        )
        # 회상한 기억을 시스템 프롬프트에 붙여 답변합니다.
        reply = model.invoke([sys] + state["messages"])
        return {"messages": [reply]}

    b = StateGraph(State)
    b.add_node("agent", agent_node)
    b.add_edge(START, "agent")
    b.add_edge("agent", END)

    # 단기(checkpointer) + 장기(store)를 함께 장착해 컴파일합니다.
    agent = b.compile(checkpointer=InMemorySaver(), store=store)
    return agent, store


st.set_page_config(page_title="기억하는 Agent", page_icon="🧠")
st.title("🧠 기억하는 Agent")
st.caption(f"모델: {MODEL} · 임베딩: {EMBED} · LO7·LO8 단기·장기 메모리 실습")


# 키가 없으면 안내만 표시하고 멈춥니다 (notebook·src와 동일한 안전 처리).
if not os.environ.get("OPENAI_API_KEY"):
    st.warning('OPENAI_API_KEY 환경변수가 필요합니다. `.env`에 키를 입력하거나 `export OPENAI_API_KEY="sk-..."` 후 다시 실행하십시오.')
    st.stop()


# 단기·장기 메모리를 담은 Agent를 1회 생성합니다 (캐시되어 rerun에도 유지).
agent, store = build_agent()


# --- 사이드바: 단기 메모리(thread_id)와 장기 메모리(Store) 가시화 ---
with st.sidebar:
    # === 단기 메모리: thread_id 선택·입력 (07_short_memory/02_checkpointer.py·03_thread_id.py에 대응) ===
    st.header("단기 메모리 (thread)")
    st.caption(
        "thread_id는 대화방 식별자입니다. 같은 thread면 직전 대화가 복원되어 맥락이 이어지고, "
        "다른 thread면 맥락이 끊깁니다."
    )
    thread_id = st.text_input(
        "thread_id",
        value="user-123",
        help="이 값을 바꾸면 다른 대화방으로 전환됩니다. 같은 값으로 돌아오면 그 방의 대화가 그대로 복원됩니다.",
    )
    if st.button("이 thread 대화 비우기", help="화면 표시용 기록만 지웁니다. (Store의 장기 기억은 그대로 유지됩니다.)"):
        st.session_state.pop(f"messages::{thread_id}", None)
        st.rerun()

    st.divider()

    # === 장기 메모리: Store 목록 표시 + 수동 추가 (08_long_memory/01_store_basics.py·02_semantic_index.py에 대응) ===
    st.header("장기 메모리 (Store)")
    st.caption(
        "장기 기억은 thread와 무관한 별도 저장소입니다. thread를 바꿔도 모든 대화가 함께 회상합니다."
    )

    # 현재 Store에 저장된 장기 기억 목록을 보여줍니다 (search로 네임스페이스 전체 조회).
    items = list(store.search(NS))
    if items:
        st.write(f"저장된 기억 {len(items)}건")
        for it in items:
            st.markdown(f"- {_text(it.value)}")
    else:
        st.write("아직 저장된 장기 기억이 없습니다.")

    # 수동으로 장기 기억을 추가합니다 (put). 추가한 기억은 새 thread에서도 회상됩니다.
    new_memory = st.text_input(
        "기억 추가",
        value="",
        placeholder="예) 앤디는 매운 음식을 못 먹는다",
        help="자유로운 문장으로 사실을 저장합니다. 시맨틱 인덱스가 텍스트를 벡터로 변환해 의미로 회상합니다.",
    )
    if st.button("Store에 저장", help="입력한 문장을 장기 기억으로 put 합니다."):
        text = new_memory.strip()
        if text:
            # key는 입력 순서대로 늘어나는 일련번호로 둡니다 (manual-1, manual-2, ...).
            key = f"manual-{len(items) + 1}"
            store.put(NS, key, {"text": text})
            st.success("장기 기억을 저장했습니다.")
            st.rerun()
        else:
            st.warning("저장할 문장을 입력해 주십시오.")


# 화면 표시용 대화 기록은 thread_id별로 분리해 세션 상태에 보관합니다.
# (실제 단기 메모리는 checkpointer가 thread_id로 관리하며, 여기서는 화면 렌더링만 담당합니다.)
display_key = f"messages::{thread_id}"
if display_key not in st.session_state:
    st.session_state[display_key] = []  # [("user"|"assistant", "텍스트"), ...]


st.caption(f"현재 thread: `{thread_id}`")

# 지난 대화를 화면에 다시 그립니다 (현재 thread의 기록만).
for role, text in st.session_state[display_key]:
    with st.chat_message(role):
        st.markdown(text)


# 사용자 입력을 받습니다.
if user_input := st.chat_input("메시지를 입력하세요"):
    # 사용자 메시지를 화면 기록에 남기고 표시합니다.
    st.session_state[display_key].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # config의 thread_id가 "어느 대화를 이어 갈지"를 결정합니다 (07_short_memory/02_checkpointer.py와 동일).
    # 같은 thread_id면 checkpointer가 직전 메시지를 복원해 맥락을 잇고, 다른 thread_id면 새 대화입니다.
    config = {"configurable": {"thread_id": thread_id}}

    with st.chat_message("assistant"):
        # invoke마다 노드가 장기 기억을 회상하고, checkpointer가 단기 맥락을 복원·누적합니다.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
        )
        answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state[display_key].append(("assistant", answer))
