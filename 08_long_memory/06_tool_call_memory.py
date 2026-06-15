"""06 - Tool-call 방식: 모델이 langmem 도구로 스스로 저장·회상한다 (모델이 자율).

이 예제 하나만으로 다음을 익힙니다.
  - langmem의 create_manage_memory_tool로 '기억하는' 도구를 만든다 (저장 담당).
  - langmem의 create_search_memory_tool로 '회상하는' 도구를 만든다 (검색 담당).
  - create_agent(v1 표준)로 두 도구·단기·장기 메모리를 한 번에 장착한다.
  - 같은 thread에서 먼저 사실을 알려 모델이 저장하게 하고, 이어 물어 회상하게 한다.

장기 기억을 Agent에 붙이는 두 방식 중 둘째입니다.
  - In-graph(05): 회상·저장의 주체 = 개발자 코드 → 제어 강함, 시점 고정
  - Tool-call(이 예제): 회상·저장의 주체 = 모델 도구 → 자율 높음, 토큰·지연 증가

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 08_long_memory/06_tool_call_memory.py

이 예제는 모델·임베딩 호출을 사용하므로 OPENAI_API_KEY가 필요합니다.
키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
# create_agent는 도구·메모리를 갖춘 Agent를 만드는 v1 표준 함수입니다.
# (구버전의 prebuilt react agent 헬퍼가 아니라, 이 create_agent를 씁니다.)
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  # 단기 메모리
from langgraph.store.memory import InMemoryStore
# langmem 도구 — 모델이 스스로 기억을 저장·검색하게 만드는 부품입니다.
from langmem import create_manage_memory_tool, create_search_memory_tool

load_dotenv()

MODEL = "openai:gpt-5.4-mini"
EMBED = "openai:text-embedding-3-small"
NS = ("user-123", "memories")


def build_tool_call_agent(store: InMemoryStore):
    """Tool-call 방식 Agent를 구성한다 (모델이 도구로 저장·검색)."""
    # 두 도구를 Agent에 붙이면, 모델이 대화 중 필요하다고 판단할 때 스스로 호출합니다.
    #   - create_manage_memory_tool: 모델이 기억을 생성/수정/삭제 (저장 담당, 도구 이름 manage_memory)
    #   - create_search_memory_tool: 모델이 자연어로 기억을 검색 (회상 담당, 도구 이름 search_memory)
    # namespace만 지정하면, 도구가 실행될 때 compile 시 넘긴 store를 런타임에서 자동으로 찾아 씁니다.
    manage_tool = create_manage_memory_tool(namespace=NS)
    search_tool = create_search_memory_tool(namespace=NS)

    # create_agent(v1 표준)로 도구·단기·장기 메모리를 한 번에 장착합니다.
    agent = create_agent(
        MODEL,
        tools=[manage_tool, search_tool],
        system_prompt=(
            "너는 사용자를 기억하는 비서다. "
            "기억할 만한 사실을 알게 되면 manage_memory 도구로 저장하고, "
            "답하기 전 필요하면 search_memory 도구로 과거 기억을 검색하라."
        ),
        checkpointer=InMemorySaver(),  # 단기 메모리
        store=store,                   # 장기 메모리 (도구가 런타임에 이 store를 사용)
    )

    # Tool-call 장단:
    #   장점(자율): 모델이 "지금 저장/검색할 때"라고 스스로 판단해 호출하므로 노드 코드가 단순합니다.
    #   단점: 호출 여부·시점을 모델이 정하므로 제어가 약하고, 토큰·지연이 늘 수 있습니다.
    return agent


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    # 시맨틱 인덱스를 켠 Store를 만듭니다 (search_memory 도구가 의미 기반으로 회상하도록).
    store = InMemoryStore(
        index={"dims": 1536, "embed": init_embeddings(EMBED), "fields": ["text"]}
    )
    agent = build_tool_call_agent(store)

    # 같은 대화 맥락을 유지하려고 thread_id를 고정합니다 (저장과 회상을 같은 thread로 잇습니다).
    cfg = {"configurable": {"thread_id": "tool-call-1"}}

    # 1) 새 사실을 알려주면 모델이 manage_memory 도구로 스스로 저장합니다.
    agent.invoke(
        {"messages": [{"role": "user", "content": "참고로 나는 주로 카페에서 일해. 기억해 둬."}]}, cfg
    )
    print("[Tool-call] 모델이 'manage_memory' 도구로 사실을 저장했습니다 (저장 주체는 모델).")

    # 2) 이어 물으면 모델이 search_memory 도구로 회상해 답합니다.
    res = agent.invoke(
        {"messages": [{"role": "user", "content": "내가 어디서 일한다고 했는지 기억나?"}]}, cfg
    )
    print("[Tool-call]", res["messages"][-1].content)  # 도구로 회상한 '카페'를 답함

    # 체크포인트: 저장도 회상도 모델이 주도해 '카페'를 답하면 Tool-call 방식을 이해한 것입니다.


if __name__ == "__main__":
    main()
