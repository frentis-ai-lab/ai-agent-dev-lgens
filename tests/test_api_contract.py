"""API 계약 테스트 — 실제 LLM 호출 없이 핵심 구성을 검증합니다.

검증 범위(키 불필요):
- v1 import 경로가 실제로 존재하는가
- @tool / Pydantic args_schema가 도구 메타데이터를 만드는가
- StateGraph가 컴파일되고 ToolNode·tools_condition이 결합되는가
- InMemorySaver / InMemoryStore의 단기·장기 메모리 기본 동작
- recursion_limit 안전망이 무한 루프를 GraphRecursionError로 끊는가

LLM·임베딩 실제 호출(model.invoke, embeddings, agent.invoke)은 키가 필요하므로
여기서 검증하지 않습니다. 강의 직전 노트북을 1회 실행해 확인합니다.
"""
from typing import Annotated
from typing_extensions import TypedDict

import pytest


def test_v1_import_paths_exist():
    # 강의 자료가 사용하는 v1 경로가 실제로 존재하는지 확인합니다.
    from langchain.chat_models import init_chat_model  # noqa: F401
    from langchain.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: F401
    from langchain.agents import create_agent  # noqa: F401
    from langchain_core.tools import tool, ToolException  # noqa: F401
    from langgraph.graph import StateGraph, START, END  # noqa: F401
    from langgraph.graph.message import add_messages  # noqa: F401
    from langgraph.prebuilt import ToolNode, tools_condition  # noqa: F401
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401
    from langgraph.store.memory import InMemoryStore  # noqa: F401
    from langchain.embeddings import init_embeddings  # noqa: F401


def test_tool_decorator_metadata():
    # LO3: @tool이 함수명·docstring·타입을 도구 메타데이터로 뽑아내는지 확인합니다.
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    assert add.name == "add"
    assert "더한다" in add.description
    assert "a" in add.args and "b" in add.args
    assert add.invoke({"a": 3, "b": 5}) == 8  # 도구 자체 실행은 LLM 불필요


def test_custom_tool_args_schema():
    # LO4: Pydantic args_schema로 입력 계약을 강제하는지 확인합니다.
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    class SearchInput(BaseModel):
        query: str = Field(description="검색어")
        top_k: int = Field(default=3, description="결과 개수")

    @tool("web_search", args_schema=SearchInput)
    def web_search(query: str, top_k: int = 3) -> str:
        """검색어로 상위 top_k건을 반환한다."""
        return f"{query} {top_k}건"

    assert web_search.name == "web_search"
    assert set(web_search.args) == {"query", "top_k"}


def test_tool_exception_raisable():
    # LO4: ToolException이 실제로 발생 가능한 예외 타입인지 확인합니다.
    from langchain_core.tools import ToolException

    with pytest.raises(ToolException):
        raise ToolException("recoverable")


def test_bind_tools_returns_runnable():
    # LO3: 모델 구성 + 도구 바인딩까지(실제 호출 없음) 동작하는지 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    model = init_chat_model("openai:gpt-5.4-mini")  # 구성만, invoke 안 함
    bound = model.bind_tools([add])
    assert hasattr(bound, "invoke")


def test_stategraph_compiles_with_toolnode():
    # LO5·LO6: StateGraph + ToolNode + tools_condition이 컴파일되는지 확인합니다.
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def call_model(state: State):
        return {"messages": []}  # 실제 모델 호출 대신 빈 갱신(구조만 검증)

    builder = StateGraph(State)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode([add]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    graph = builder.compile()
    assert graph is not None


def test_short_term_memory_store():
    # LO7·LO8: 단기(InMemorySaver) 구성 + 장기(InMemoryStore) put/get 기본 동작.
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore

    assert InMemorySaver() is not None

    store = InMemoryStore()  # 시맨틱 인덱스 없이도 put/get은 동작(임베딩 불필요)
    ns = ("user-123", "memories")
    store.put(ns, "fact-1", {"text": "앤디는 파이썬을 좋아한다"})
    assert store.get(ns, "fact-1").value["text"] == "앤디는 파이썬을 좋아한다"
    store.delete(ns, "fact-1")
    assert store.get(ns, "fact-1") is None


def test_recursion_limit_stops_infinite_loop():
    # LO5·LO6: recursion_limit 안전망이 무한 루프를 GraphRecursionError로 끊는지 확인합니다.
    from langgraph.graph import StateGraph, START, END
    from langgraph.errors import GraphRecursionError

    class State(TypedDict):
        n: int

    def step(state: State):
        return {"n": state["n"] + 1}

    def route(state: State) -> str:
        return "loop" if state["n"] < 100 else "end"

    builder = StateGraph(State)
    builder.add_node("loop", step)
    builder.add_edge(START, "loop")
    builder.add_conditional_edges("loop", route, {"loop": "loop", "end": END})
    graph = builder.compile()

    # 명시적으로 낮춘 한도(4)를 넘으면 무한히 돌지 않고 예외로 멈춰야 합니다.
    with pytest.raises(GraphRecursionError):
        graph.invoke({"n": 0}, {"recursion_limit": 4})
