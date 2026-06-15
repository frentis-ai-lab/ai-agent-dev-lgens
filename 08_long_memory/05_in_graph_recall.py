"""05 - In-graph 방식: 그래프 노드가 직접 store.search로 회상한다 (개발자가 제어).

이 예제 하나만으로 다음을 익힙니다.
  - 그래프 노드가 store.search(...)를 직접 불러 장기 기억을 회상한다.
  - 회상한 기억을 SystemMessage에 끼워 넣어 모델이 답할 때 참고하게 한다.
  - compile(store=...)로 넘긴 Store가 노드 인자 (*, store)로 자동 주입된다.
  - 회상 값을 '방어적으로' 읽는다 (직접 put한 dict와 도구가 저장한 dict가 섞일 수 있음).

장기 기억을 Agent에 붙이는 두 방식 중 첫째입니다.
  - In-graph(이 예제): 회상·저장의 주체 = 개발자 코드 → 제어 강함, 시점 고정
  - Tool-call(06):    회상·저장의 주체 = 모델 도구 → 자율 높음, 토큰·지연 증가

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 08_long_memory/05_in_graph_recall.py

이 예제는 모델·임베딩 호출을 사용하므로 OPENAI_API_KEY가 필요합니다.
키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os
# typing은 타입을 표현하는 도구 모음입니다. Annotated[타입, 부가정보]로 칸에 동작을 덧붙입니다.
from typing import Annotated

# TypedDict는 "어떤 키와 타입을 갖는 dict"인지 미리 선언하는 틀입니다.
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
# SystemMessage는 모델에게 역할·규칙을 주는 메시지입니다 (v1 권장 경로).
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver  # 단기 메모리(checkpointer)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# BaseStore는 노드 인자에 주입되는 Store의 공통 타입입니다 (타입 힌트용).
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

load_dotenv()

MODEL = "openai:gpt-5.4-mini"
EMBED = "openai:text-embedding-3-small"
NS = ("user-123", "memories")


# State는 그래프가 한 턴마다 주고받는 상태입니다. messages 한 칸만 두고 add_messages로 누적합니다.
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 단기: 메시지 누적


def build_in_graph_agent(store: InMemoryStore):
    """In-graph 방식 Agent를 구성한다 (노드가 직접 회상)."""
    model = init_chat_model(MODEL)

    # 노드가 Store를 직접 호출합니다. compile 시 넘긴 store가 인자 (*, store)로 자동 주입됩니다.
    # (*, store)에서 *는 "store는 키워드로만 받는다"는 표시이며, LangGraph가 알아서 채워 줍니다.
    def agent_node(state: State, *, store: BaseStore):
        last = state["messages"][-1].content
        # 1) 회상: 마지막 발화와 의미가 가까운 장기 기억을 상위 3개만 가져옵니다 (개발자가 직접 제어).
        recalled = store.search(NS, query=last, limit=3)

        # 같은 Store에 직접 put한 {"text": ...}와, langmem 도구가 저장한 형태({"content": ...} 등)가
        # 섞일 수 있으므로, 값에서 텍스트를 안전하게 꺼냅니다 (text → content → 통째로 문자열화).
        # 주의: 이 방어적 읽기를 r.value['text'] 같은 직접 접근으로 되돌리면, 도구가 저장한
        #      항목에는 'text' 키가 없어 KeyError가 납니다. 직접 접근으로 바꾸지 마십시오.
        def _text(value):
            if isinstance(value, dict):
                return value.get("text") or value.get("content") or str(value)
            return str(value)

        # "\n".join(...)은 여러 줄을 줄바꿈으로 이어 붙입니다. or "(없음)"은 회상이 비면 대체 문구입니다.
        memory_text = "\n".join(f"- {_text(r.value)}" for r in recalled) or "(없음)"
        sys = SystemMessage(
            "너는 사용자를 기억하는 비서다. 아래는 회상한 사실이며, "
            "질문과 어긋나면 무시하라.\n" + memory_text
        )
        # 2) 응답: 회상한 기억을 시스템 프롬프트에 끼워 답변합니다.
        reply = model.invoke([sys] + state["messages"])
        return {"messages": [reply]}

    b = StateGraph(State)
    b.add_node("agent", agent_node)
    b.add_edge(START, "agent")
    b.add_edge("agent", END)

    # 단기(checkpointer) + 장기(store)를 함께 장착합니다.
    agent = b.compile(checkpointer=InMemorySaver(), store=store)

    # In-graph 장단:
    #   장점(제어): 언제·무엇을·몇 개 회상할지, 어떻게 프롬프트에 넣을지 코드로 100% 결정합니다.
    #   단점: 저장·회상 시점이 코드에 고정되어, 모델이 "지금 기억할 만하다"고 스스로 판단하지는 못합니다.
    return agent


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    # 시맨틱 인덱스를 켠 Store를 만들고, 회상할 기억 한 건을 미리 심어 둡니다.
    store = InMemoryStore(
        index={"dims": 1536, "embed": init_embeddings(EMBED), "fields": ["text"]}
    )
    store.put(NS, "seed", {"text": "앤디는 야간 근무를 선호한다"})

    in_graph = build_in_graph_agent(store)
    res = in_graph.invoke(
        {"messages": [{"role": "user", "content": "내가 선호하는 근무 시간대가 뭐였지?"}]},
        {"configurable": {"thread_id": "in-graph-1"}},
    )
    print("[In-graph]", res["messages"][-1].content)  # 노드가 seed를 회상해 '야간'을 답함

    # 체크포인트: 코드가 직접 search한 seed 기억으로 '야간'을 답하면 In-graph 회상을 이해한 것입니다.


if __name__ == "__main__":
    main()
