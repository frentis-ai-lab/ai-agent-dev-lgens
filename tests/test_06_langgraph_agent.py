"""06_langgraph_agent 챕터 테스트 — 계약 테스트 + 실제 키 게이트 테스트.

이 파일은 두 묶음으로 나뉩니다.

1) 키 없이 도는 '계약 테스트' (대부분)
   - 각 예제 파일이 존재·py_compile·import 되는지, 짝 .md가 있는지 확인합니다.
   - 옛 단일 파일(lab.py)이 삭제되어 파일별 예제 구조로 재편됐는지 확인합니다.
   - LangGraph/LangChain v1 Agent 핵심 API 계약을 키 없이 검증합니다:
       * StateGraph + ToolNode + tools_condition로 수동 Agent 그래프가 compile()되는지
       * create_agent가 invoke·stream을 가진 컴파일된 그래프를 돌려주는지
       * ToolNode가 handle_tool_errors로 도구 예외를 잡아 ToolMessage로 돌려주는지
       * GraphRecursionError를 import할 수 있고, 종료 조건 없는 순환이 한도 초과로 멈추는지
   - conftest.py가 더미 키(OPENAI_API_KEY)를 넣어 두므로 모델 객체 구성 단계가
     키 부재로 막히지 않습니다. 모델을 부르는 그래프는 이 묶음에서 invoke하지 않습니다.

2) 실제 키가 있을 때만 도는 '실제 호출 테스트'
   - @pytest.mark.skipif로, 진짜 키가 없으면 건너뜁니다(skip).
   - 키가 있으면 create_agent로 만든 Agent를 1회 실제로 invoke해, 도구를 거쳐
     최종 답(32)이 messages에 누적되는지 최소 검증합니다.

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
CHAPTER_DIR = REPO_ROOT / "06_langgraph_agent"

# 챕터의 6개 예제 파일. 번호 순서대로 둡니다.
EXAMPLE_FILES = [
    "01_manual_agent_graph.py",
    "02_react_loop_observe.py",
    "03_create_agent.py",
    "04_multi_tool_agent.py",
    "05_custom_state.py",
    "06_error_and_safety.py",
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
    mod_name = "lg_agent_" + filename.replace(".py", "")
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


def test_chapter_readme_exists():
    # 챕터 README가 존재하는지 확인합니다.
    assert (CHAPTER_DIR / "README.md").is_file(), "챕터 README.md가 보이지 않습니다."


@pytest.mark.parametrize("filename", EXAMPLE_FILES + ["README.md"])
def test_no_create_react_agent(filename):
    # 이 챕터는 v1 권장 진입점 create_agent만 씁니다. create_react_agent는 금지입니다.
    text = (CHAPTER_DIR / filename).read_text(encoding="utf-8")
    assert "create_react_agent" not in text, f"{filename} 에 금지된 create_react_agent가 있습니다."


@pytest.mark.parametrize("filename", ["03_create_agent.md", "04_multi_tool_agent.md",
                                      "05_custom_state.md", "06_error_and_safety.md"])
def test_readme_has_mermaid(filename):
    # 새 예제의 학습 문서마다 mermaid 다이어그램이 포함됐는지 확인합니다.
    text = (CHAPTER_DIR / filename).read_text(encoding="utf-8")
    assert "```mermaid" in text, f"{filename} 에 mermaid 다이어그램이 없습니다."


# --------------------------------------------------------------------------
# 1) 키 없이 도는 계약 테스트 — v1 Agent / LangGraph 핵심 API
# --------------------------------------------------------------------------

def test_manual_agent_graph_compiles_with_toolnode():
    # 01 예제의 핵심: StateGraph + ToolNode + tools_condition + 되돌아오는 엣지로
    # 수동 Agent 그래프가 compile()되어 invoke 가능한 그래프가 되는지(모델 없이) 확인합니다.
    from langgraph.graph import StateGraph, START
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def call_model(state: State) -> dict:
        return {}  # 모델 없이 노드 자리만 채웁니다(컴파일·배선만 검증).

    builder = StateGraph(State)
    builder.add_node("model", call_model)
    # ToolNode는 도구 목록만으로 구성됩니다(키·모델 불필요).
    builder.add_node("tools", ToolNode([]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    graph = builder.compile()

    # 컴파일된 그래프는 invoke를 가진 실행 단위입니다.
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "stream")


def test_create_agent_returns_compiled_graph():
    # 03~05 예제의 핵심: create_agent가 invoke·stream을 가진 컴파일된 그래프를 돌려주는지
    # (모델 호출 없이, 그래프 구성만) 확인합니다.
    from langchain.agents import create_agent
    from langchain.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    # 더미 키(conftest)만으로 그래프 구성은 됩니다. 실제 모델 호출은 하지 않습니다.
    agent = create_agent(
        "openai:gpt-5.4-mini",
        tools=[add],
        system_prompt="너는 계산 비서다.",
    )
    assert hasattr(agent, "invoke"), "create_agent가 invoke 가진 그래프를 돌려줘야 합니다."
    assert hasattr(agent, "stream"), "create_agent가 stream 가진 그래프를 돌려줘야 합니다."


def test_create_agent_with_custom_state_schema():
    # 05 예제의 계약: AgentState를 상속한 커스텀 상태와 동적 프롬프트 미들웨어로
    # create_agent가 그래프를 구성하는지(모델 호출 없이) 확인합니다.
    from langchain.agents import create_agent, AgentState
    from langchain.agents.middleware import dynamic_prompt, ModelRequest

    class SupportState(AgentState):
        user_name: str
        tier: str

    @dynamic_prompt
    def support_prompt(request: ModelRequest) -> str:
        return f"너는 비서다. {request.state.get('user_name', '고객')}님을 응대하라."

    agent = create_agent(
        "openai:gpt-5.4-mini",
        tools=[],
        state_schema=SupportState,
        middleware=[support_prompt],
    )
    assert hasattr(agent, "invoke")


def test_toolnode_handle_tool_errors_returns_toolmessage():
    # 06 예제의 핵심: ToolNode가 handle_tool_errors=True(기본)에서 도구 예외를 잡아
    # 오류 메시지를 담은 ToolMessage로 돌려주는지(모델 없이) 확인합니다.
    # ToolNode는 그래프 런타임 위에서 동작하므로, 최소 그래프에 넣어 invoke합니다.
    from langchain.tools import tool
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    @tool
    def divide(a: int, b: int) -> int:
        """두 정수를 나눈다."""
        return a // b  # b=0이면 ZeroDivisionError

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    b = StateGraph(State)
    b.add_node("tools", ToolNode([divide], handle_tool_errors=True))
    b.add_edge(START, "tools")
    b.add_edge("tools", END)
    graph = b.compile()

    # 모델이 0으로 나누는 도구 호출을 요청한 상황을 손으로 만들어 그래프에 넘깁니다.
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "divide", "args": {"a": 10, "b": 0}, "id": "call_1", "type": "tool_call"}],
    )
    result = graph.invoke({"messages": [ai]})
    # 예외가 그래프를 죽이지 않고, 오류를 담은 ToolMessage로 돌아와야 합니다.
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    # 오류 내용이 모델이 읽을 수 있는 문자열로 담깁니다.
    assert isinstance(tool_messages[0].content, str) and tool_messages[0].content.strip()


def test_graph_recursion_error_importable():
    # 06 예제의 계약: GraphRecursionError를 v1 권장 경로에서 import할 수 있어야 합니다.
    from langgraph.errors import GraphRecursionError

    assert issubclass(GraphRecursionError, Exception)


def test_endless_loop_raises_recursion_error():
    # 06 예제의 안전망 계약: 종료 조건 없는 순환은 recursion_limit 초과로 GraphRecursionError를 냅니다.
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


# --------------------------------------------------------------------------
# 2) 실제 키가 있을 때만 도는 실제 호출 테스트
#    진짜 키가 없으면 skip 됩니다(더미 키로는 호출하지 않습니다).
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_create_agent_invoke_live():
    # create_agent로 만든 Agent를 1회 실제로 invoke해, 도구를 거쳐 최종 답(32)이
    # messages에 누적되는지 확인합니다.
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_core.messages import AIMessage

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """두 정수를 곱한다."""
        return a * b

    agent = create_agent(
        "openai:gpt-5.4-mini",
        tools=[add, multiply],
        system_prompt="너는 정확한 계산 비서다. 계산은 반드시 도구로 수행하라.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "3 더하기 5를 4와 곱하면?"}]}
    )
    messages = result["messages"]
    # 입력 + 도구 호출/결과 + 최종 답까지 여러 메시지가 누적되어야 합니다.
    assert len(messages) >= 2
    last = messages[-1]
    assert isinstance(last, AIMessage)
    assert isinstance(last.content, str) and last.content.strip()
    # 두 도구를 거친 최종 계산 결과 32가 답에 담겨야 합니다.
    assert "32" in last.content
