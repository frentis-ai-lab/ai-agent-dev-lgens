"""05_langgraph_workflow 챕터 테스트 — 계약 테스트 + 실제 키 게이트 테스트.

이 파일은 두 묶음으로 나뉩니다.

1) 키 없이 도는 '계약 테스트' (대부분)
   - 각 예제 파일이 존재·py_compile·import 되는지 확인합니다.
   - LangGraph 핵심 API 계약을 키 없이 검증합니다:
       * StateGraph가 노드·엣지로 조립되어 compile()로 실행 가능한 그래프가 되는지
       * add_messages 리듀서가 메시지를 덮어쓰지 않고 누적하는지
       * 리듀서가 없으면 기본 동작이 덮어쓰기인지
       * add_conditional_edges로 라우터 분기가 구성되고, 모델 없는 그래프가 실제로 분기하는지
       * GraphRecursionError를 import할 수 있고, 종료 조건 없는 순환이 한도 초과로 멈추는지
   - conftest.py가 더미 키(OPENAI_API_KEY)를 넣어 두므로 모델 객체 구성 단계가
     키 부재로 막히지 않습니다. 모델을 부르는 그래프는 이 묶음에서 invoke하지 않습니다.

2) 실제 키가 있을 때만 도는 '실제 호출 테스트'
   - @pytest.mark.skipif로, 진짜 키가 없으면 건너뜁니다(skip).
   - 키가 있으면 모델 노드 그래프를 1회 실제로 invoke해, AIMessage가 messages에
     누적되는지 최소 검증합니다.

실제 호출 테스트는 강의 직전에 강사가 키를 넣고 1회 돌려 확인하는 용도입니다.
"""
import importlib.util
import operator
import os
import py_compile
from pathlib import Path
from typing import Annotated

import pytest
from typing_extensions import TypedDict

# 이 테스트 파일(tests/)의 부모가 레포 루트(ai-agent-dev-lgens)입니다.
REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIR = REPO_ROOT / "05_langgraph_workflow"

# 챕터의 6개 예제 파일. 번호 순서대로 둡니다.
EXAMPLE_FILES = [
    "01_state_and_graph.py",
    "02_nodes_and_edges.py",
    "03_reducers.py",
    "04_conditional_edge.py",
    "05_router_patterns.py",
    "06_loop_and_recursion.py",
]

# conftest.py가 setdefault로 넣는 더미 키 값. 이 값이면 '진짜 키 없음'으로 봅니다.
_DUMMY_OPENAI_KEY = "sk-test-dummy-not-used"


def _has_real_openai_key() -> bool:
    """conftest의 더미 키가 아니라 진짜 OPENAI_API_KEY가 있는지 판별합니다."""
    key = os.getenv("OPENAI_API_KEY")
    return bool(key) and key != _DUMMY_OPENAI_KEY


def _load_module(filename: str):
    """파일 경로로 모듈을 직접 로드합니다(숫자로 시작하는 파일명은 일반 import가 안 됨).

    파일 상단의 import·load_dotenv·MODEL 상수까지 실제로 실행되지만,
    main()은 `if __name__ == "__main__"` 안에 있어 자동 실행되지 않습니다.
    따라서 모델 호출 없이 모듈 정의만 메모리에 올라옵니다.
    """
    path = CHAPTER_DIR / filename
    # 모듈 이름은 충돌을 피하려 파일명 앞에 접두사를 붙입니다.
    mod_name = "lg_wf_" + filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1) 키 없이 도는 계약 테스트 — 파일·컴파일·import
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_files_exist(filename):
    # 6개 예제 파일이 모두 챕터 폴더에 실제로 존재하는지 확인합니다.
    assert (CHAPTER_DIR / filename).is_file(), f"{filename} 이 보이지 않습니다."


def test_lab_py_removed():
    # 옛 단일 파일(lab.py)이 삭제되어 파일별 예제 구조로 재편되었는지 확인합니다.
    assert not (CHAPTER_DIR / "lab.py").exists(), "lab.py가 아직 남아 있습니다(삭제 대상)."


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_files_compile(filename):
    # 각 예제 파일이 문법 오류 없이 컴파일되는지 확인합니다(키 불필요).
    py_compile.compile(str(CHAPTER_DIR / filename), doraise=True)


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_files_import(filename):
    # 각 예제 파일이 import 가능한지(상단 import·MODEL 상수까지 실행) 확인합니다.
    # main()은 자동 실행되지 않으므로 모델 호출은 일어나지 않습니다.
    module = _load_module(filename)
    # 모든 예제는 main 함수와 MODEL 상수를 가집니다.
    assert hasattr(module, "main"), f"{filename} 에 main()이 없습니다."
    assert module.MODEL == "openai:gpt-5.4-mini"


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_has_readme(filename):
    # 각 예제 파일에 짝이 되는 학습 문서(.md)가 함께 있는지 확인합니다.
    md = filename.replace(".py", ".md")
    assert (CHAPTER_DIR / md).is_file(), f"{md} 짝 문서가 보이지 않습니다."


# --------------------------------------------------------------------------
# 1) 키 없이 도는 계약 테스트 — LangGraph 핵심 API
# --------------------------------------------------------------------------

def test_stategraph_compiles_and_invokes_without_model():
    # 01 예제의 핵심: StateGraph가 노드·엣지로 조립되어 compile() 뒤 invoke되는지
    # (모델 없이) 확인합니다.
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        topic: str
        result: str

    def echo(state: State) -> dict:
        return {"result": f"입력받은 글감: {state['topic']}"}

    builder = StateGraph(State)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    graph = builder.compile()

    # 컴파일된 그래프는 invoke를 가진 실행 단위입니다.
    assert hasattr(graph, "invoke")
    result = graph.invoke({"topic": "LangGraph", "result": ""})
    # 노드가 채운 result 칸이 반영되고, 손대지 않은 topic 칸은 그대로 남습니다.
    assert result["result"] == "입력받은 글감: LangGraph"
    assert result["topic"] == "LangGraph"


def test_compile_required_before_invoke():
    # 빌더 자체는 invoke 인터페이스를 갖지 않습니다. compile()해야 실행 가능해집니다.
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        x: int

    def noop(state: State) -> dict:
        return {}

    builder = StateGraph(State)
    builder.add_node("noop", noop)
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)

    # 컴파일 전 빌더에는 invoke가 없습니다(컴파일이 실행의 전제임을 계약으로 고정).
    assert not hasattr(builder, "invoke")
    graph = builder.compile()
    assert hasattr(graph, "invoke")


def test_add_messages_reducer_accumulates():
    # 03 예제의 핵심: Annotated[list, add_messages]를 붙이면 메시지가 누적됩니다.
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def add_one(state: State) -> dict:
        return {"messages": [AIMessage("누적했습니다")]}

    b = StateGraph(State)
    b.add_node("add_one", add_one)
    b.add_edge(START, "add_one")
    b.add_edge("add_one", END)
    graph = b.compile()

    result = graph.invoke({"messages": [HumanMessage("입력")]})
    # 입력 1 + 새 메시지 1 = 2 (덮어쓰기였다면 1)
    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "입력"
    assert result["messages"][-1].content == "누적했습니다"


def test_no_reducer_overwrites_by_default():
    # 03 예제의 대비: 리듀서가 없는 list 칸은 기본 동작이 '덮어쓰기'입니다.
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        messages: list  # add_messages가 없으므로 덮어쓰기

    def overwrite(state: State) -> dict:
        return {"messages": [AIMessage("덮어썼습니다")]}

    b = StateGraph(State)
    b.add_node("overwrite", overwrite)
    b.add_edge(START, "overwrite")
    b.add_edge("overwrite", END)
    graph = b.compile()

    result = graph.invoke({"messages": [HumanMessage("이 입력은 사라집니다")]})
    # 입력이 덮어써져 사라지고 1개만 남습니다.
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "덮어썼습니다"


def test_operator_add_reducer_sums():
    # 03 예제의 일반화: operator.add 리듀서는 숫자 칸을 합산합니다.
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        total: Annotated[int, operator.add]

    def add_three(state: State) -> dict:
        return {"total": 3}

    b = StateGraph(State)
    b.add_node("add_three", add_three)
    b.add_edge(START, "add_three")
    b.add_edge("add_three", END)
    graph = b.compile()

    # 시작 10 + 노드 반환 3 = 13 (덮어쓰기였다면 3)
    assert graph.invoke({"total": 10})["total"] == 13


def test_add_conditional_edges_branches_without_model():
    # 04 예제의 핵심: add_conditional_edges로 라우터 분기가 구성되고,
    # 모델 없는 그래프가 입력에 따라 실제로 다른 경로로 가는지 확인합니다.
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        text: str
        path: str

    def classify(state: State) -> dict:
        return {}  # 통과 노드

    def route(state: State) -> str:
        # 길이 임계치로 분기 키를 정합니다(모델 없이 분기 동작만 검증).
        return "long" if len(state["text"]) > 5 else "short"

    def long_node(state: State) -> dict:
        return {"path": "long"}

    def short_node(state: State) -> dict:
        return {"path": "short"}

    b = StateGraph(State)
    b.add_node("classify", classify)
    b.add_node("long", long_node)
    b.add_node("short", short_node)
    b.add_edge(START, "classify")
    b.add_conditional_edges("classify", route, {"long": "long", "short": "short"})
    b.add_edge("long", END)
    b.add_edge("short", END)
    graph = b.compile()

    assert graph.invoke({"text": "안녕", "path": ""})["path"] == "short"
    assert graph.invoke({"text": "오늘 회의 내용을 정리했다", "path": ""})["path"] == "long"


def test_graph_recursion_error_importable():
    # 06 예제의 계약: GraphRecursionError를 v1 권장 경로에서 import할 수 있어야 합니다.
    from langgraph.errors import GraphRecursionError

    # 예외 클래스인지(Exception을 상속하는지) 확인합니다.
    assert issubclass(GraphRecursionError, Exception)


def test_endless_loop_raises_recursion_error():
    # 06 예제의 핵심: 종료 조건 없는 순환은 recursion_limit 초과로 GraphRecursionError를 냅니다.
    from langgraph.graph import StateGraph, START, END
    from langgraph.errors import GraphRecursionError

    class CounterState(TypedDict):
        step: Annotated[int, operator.add]

    def tick(state: CounterState) -> dict:
        return {"step": 1}

    def loop_route(state: CounterState) -> str:
        return "loop"  # 항상 자기 자신으로 되돌림(종료 조건 없음)

    b = StateGraph(CounterState)
    b.add_node("tick", tick)
    b.add_edge(START, "tick")
    b.add_conditional_edges("tick", loop_route, {"loop": "tick", "end": END})
    graph = b.compile()

    # 한도를 낮춰 무한 루프 대신 한도 초과로 안전하게 멈추는지 확인합니다.
    with pytest.raises(GraphRecursionError):
        graph.invoke({"step": 0}, {"recursion_limit": 5})


def test_loop_with_exit_condition_terminates():
    # 06 예제의 근본 해결: 라우터에 종료 조건을 두면 한도에 닿기 전에 정상 종료합니다.
    from langgraph.graph import StateGraph, START, END

    class CounterState(TypedDict):
        step: Annotated[int, operator.add]

    def tick(state: CounterState) -> dict:
        return {"step": 1}

    def loop_route(state: CounterState) -> str:
        return "end" if state["step"] >= 3 else "loop"

    b = StateGraph(CounterState)
    b.add_node("tick", tick)
    b.add_edge(START, "tick")
    b.add_conditional_edges("tick", loop_route, {"loop": "tick", "end": END})
    graph = b.compile()

    # 종료 조건이 있으니 넉넉한 한도에서도 step=3에서 스스로 멈춥니다.
    result = graph.invoke({"step": 0}, {"recursion_limit": 100})
    assert result["step"] == 3


# --------------------------------------------------------------------------
# 2) 실제 키가 있을 때만 도는 실제 호출 테스트
#    진짜 키가 없으면 skip 됩니다(더미 키로는 호출하지 않습니다).
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_model_node_graph_invoke_live():
    # 모델을 부르는 노드 그래프를 1회 실제로 invoke해, AIMessage가 messages에 누적되는지 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    model = init_chat_model("openai:gpt-5.4-mini")

    def chatbot(state: State) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    b = StateGraph(State)
    b.add_node("chatbot", chatbot)
    b.add_edge(START, "chatbot")
    b.add_edge("chatbot", END)
    graph = b.compile()

    result = graph.invoke(
        {"messages": [HumanMessage("한 단어로만 답해: 대한민국의 수도는?")]}
    )
    # 입력 메시지 뒤에 모델 응답(AIMessage)이 누적되어야 합니다.
    assert len(result["messages"]) >= 2
    assert isinstance(result["messages"][-1], AIMessage)
    assert isinstance(result["messages"][-1].content, str) and result["messages"][-1].content.strip()
