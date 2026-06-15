"""07_short_memory 챕터 테스트 — 계약 테스트 + 실제 키 게이트 테스트.

이 파일은 두 묶음으로 나뉩니다.

1) 키 없이 도는 '계약 테스트' (대부분)
   - 각 예제 파일이 존재·py_compile·import 되는지, 짝 README가 있는지 확인합니다.
   - 옛 단일 파일(lab.py)이 삭제되어 파일별 예제 구조로 재편되었는지 확인합니다.
   - 단기 메모리 핵심 API 계약을 키 없이 검증합니다:
       * InMemorySaver를 구성하고 create_agent에 checkpointer로 붙여 컴파일되는지
       * checkpointer를 붙인 에이전트가 get_state·get_state_history 인터페이스를 갖는지
       * trim_messages가 토큰 상한에 맞춰 자르고 SystemMessage를 보존하는지
       * SummarizationMiddleware를 구성해 에이전트에 끼울 수 있는지
   - conftest.py가 더미 키(OPENAI_API_KEY)를 넣어 두므로 모델 객체 구성 단계가
     키 부재로 막히지 않습니다. 모델을 부르는 에이전트는 이 묶음에서 invoke하지 않습니다.

2) 실제 키가 있을 때만 도는 '실제 호출 테스트'
   - @pytest.mark.skipif로, 진짜 키가 없으면 건너뜁니다(skip).
   - 키가 있으면 같은 thread_id로는 이름을 기억하고, 다른 thread_id로는 기억이 격리되는지
     최소로 검증합니다.

실제 호출 테스트는 강의 직전에 강사가 키를 넣고 1회 돌려 확인하는 용도입니다.
"""
import importlib.util
import os
import py_compile
from pathlib import Path

import pytest

# 이 테스트 파일(tests/)의 부모가 레포 루트(ai-agent-dev-lgens)입니다.
REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIR = REPO_ROOT / "07_short_memory"

# 챕터의 6개 예제 파일. 번호 순서대로 둡니다.
EXAMPLE_FILES = [
    "01_no_memory.py",
    "02_checkpointer.py",
    "03_thread_id.py",
    "04_inspect_state.py",
    "05_trim_messages.py",
    "06_summarize_persist.py",
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
    mod_name = "short_mem_" + filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1) 키 없이 도는 계약 테스트 — 파일·컴파일·import·문서
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
    # 챕터 README가 있는지 확인합니다.
    assert (CHAPTER_DIR / "README.md").is_file(), "챕터 README.md가 보이지 않습니다."


# --------------------------------------------------------------------------
# 1) 키 없이 도는 계약 테스트 — 단기 메모리 핵심 API
# --------------------------------------------------------------------------

def test_inmemorysaver_constructs_from_v1_path():
    # 02 예제의 계약: InMemorySaver를 v1 권장 경로에서 가져와 구성할 수 있어야 합니다.
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    # checkpointer는 상태를 읽고 쓰는 인터페이스(get_tuple·put)를 갖춘 객체입니다.
    assert hasattr(checkpointer, "get_tuple")
    assert hasattr(checkpointer, "put")


def test_create_agent_with_checkpointer_compiles_without_model_call():
    # 02 예제의 핵심: create_agent에 checkpointer를 붙이면 실행 가능한 그래프로 컴파일되고,
    # 상태 조회 인터페이스(get_state·get_state_history)를 갖추는지 (호출 없이) 확인합니다.
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    # conftest의 더미 키로 모델 객체 구성까지만 합니다(invoke 안 함).
    agent = create_agent(
        "openai:gpt-5.4-mini",
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        checkpointer=InMemorySaver(),
    )
    # 컴파일된 에이전트는 invoke·get_state·get_state_history를 가진 실행 단위입니다.
    assert hasattr(agent, "invoke")
    assert hasattr(agent, "get_state")
    assert hasattr(agent, "get_state_history")


def test_config_thread_id_shape():
    # 02~04 예제의 함정: thread_id는 입력 메시지가 아니라 configurable 설정으로 넘깁니다.
    # 형식이 정확한지(중첩 키 위치)를 계약으로 고정합니다.
    config = {"configurable": {"thread_id": "user-123"}}
    assert "configurable" in config
    assert config["configurable"]["thread_id"] == "user-123"


def test_make_thread_id_combines_keys():
    # 03 예제의 핵심: 세션 키처럼 사용자 ID와 대화방 ID를 조합한 합성 thread_id가 만들어지는지 확인합니다.
    module = _load_module("03_thread_id.py")
    # 같은 사용자라도 대화방이 다르면 thread_id가 달라져 대화가 격리됩니다.
    assert module.make_thread_id("emp-042", "room-7") == "emp-042:room-7"
    assert module.make_thread_id("emp-042", "room-7") != module.make_thread_id("emp-042", "room-9")


def test_trim_messages_cuts_to_token_budget_and_keeps_system():
    # 05 예제의 핵심: trim_messages가 토큰 상한 안으로 자르되 SystemMessage를 보존하고,
    # 최근 대화를 우선 남기는지(모델 호출 없이) 확인합니다.
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        trim_messages,
    )
    from langchain_core.messages.utils import count_tokens_approximately

    # 긴 대화를 손으로 만듭니다 (시스템 1개 + 사용자·답변 5쌍 = 11개).
    messages = [SystemMessage("너는 친절한 한국어 비서다.")]
    for i in range(5):
        messages.append(HumanMessage(f"{i}번째 질문입니다. " * 8))
        messages.append(AIMessage(f"{i}번째 답변입니다. " * 8))

    before = count_tokens_approximately(messages)

    trimmed = trim_messages(
        messages,
        max_tokens=120,
        token_counter=count_tokens_approximately,
        strategy="last",
        include_system=True,
        start_on="human",
    )
    after = count_tokens_approximately(trimmed)

    # 자른 뒤 토큰이 상한 이하로 줄고, 원본보다 작아야 합니다.
    assert after <= 120
    assert after < before
    assert len(trimmed) < len(messages)  # 메시지 수도 줄어듭니다
    # 역할 지시인 SystemMessage는 보존되어야 합니다.
    assert any(isinstance(m, SystemMessage) for m in trimmed)
    # 최근 대화를 우선 남기므로(strategy="last"), 마지막 메시지는 원본 마지막과 같아야 합니다.
    assert trimmed[-1].content == messages[-1].content


def test_trim_messages_helper_in_example_makes_long_conversation():
    # 05 예제의 보조 함수가 11개 메시지(시스템 1 + 5쌍)를 만드는지 확인합니다.
    module = _load_module("05_trim_messages.py")
    messages = module.make_long_conversation()
    assert len(messages) == 11
    from langchain_core.messages import SystemMessage
    assert isinstance(messages[0], SystemMessage)  # 맨 앞은 시스템 메시지


def test_summarization_middleware_constructs():
    # 06 예제의 계약: SummarizationMiddleware를 v1 경로에서 가져와 trigger·keep으로 구성할 수 있어야 합니다.
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    summarizer = SummarizationMiddleware(
        model="openai:gpt-5.4-mini",
        trigger=("messages", 6),
        keep=("messages", 4),
    )

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    # 요약 미들웨어를 끼운 에이전트가 컴파일되는지(호출 없이) 확인합니다.
    agent = create_agent(
        "openai:gpt-5.4-mini",
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        checkpointer=InMemorySaver(),
        middleware=[summarizer],
    )
    assert hasattr(agent, "invoke")


# --------------------------------------------------------------------------
# 2) 실제 키가 있을 때만 도는 실제 호출 테스트
#    진짜 키가 없으면 skip 됩니다(더미 키로는 호출하지 않습니다).
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_same_thread_remembers_live():
    # 같은 thread_id로 두 턴을 주고받으면 앞 대화(이름)를 기억하고 메시지가 누적되는지 확인합니다.
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    agent = create_agent(
        "openai:gpt-5.4-mini",
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "test-same"}}

    agent.invoke({"messages": [{"role": "user", "content": "내 이름은 앤디야. 기억해 줘."}]}, config)
    r2 = agent.invoke({"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]}, config)

    # 같은 thread이므로 앞 대화가 복원되어 이름을 답에 담아야 합니다.
    assert "앤디" in r2["messages"][-1].content
    # 두 턴(user·ai)이 누적되어 메시지가 4개 이상이어야 합니다.
    assert len(r2["messages"]) >= 4


@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_other_thread_breaks_context_live():
    # thread_id를 바꾸면 앞 대화의 기억이 격리되어, 이름을 답에 담지 못하는지 확인합니다.
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    agent = create_agent(
        "openai:gpt-5.4-mini",
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        checkpointer=InMemorySaver(),
    )

    # thread A에 이름을 저장합니다.
    config_a = {"configurable": {"thread_id": "test-A"}}
    agent.invoke({"messages": [{"role": "user", "content": "내 이름은 앤디야. 기억해 줘."}]}, config_a)

    # thread B로 바꿔 물으면 A의 기억이 없으므로 이름을 답에 담지 못해야 합니다(의도된 격리).
    config_b = {"configurable": {"thread_id": "test-B"}}
    r = agent.invoke({"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]}, config_b)
    assert "앤디" not in r["messages"][-1].content
