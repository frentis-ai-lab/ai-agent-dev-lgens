"""03_tool_calling 챕터 테스트 — 계약 테스트 + 실제 키 게이트 테스트.

이 파일은 두 묶음으로 나뉩니다.

1) 키 없이 도는 '계약 테스트' (대부분)
   - 각 예제 파일이 존재·py_compile·import 되는지 확인합니다.
   - 핵심 구성(@tool이 도구 메타데이터를 만듦, bind_tools가 Runnable을 반환,
     tool_choice·parallel_tool_calls가 결합됨, ToolException이 던질 수 있음)을
     키 없이 검증합니다.
   - conftest.py가 더미 키(OPENAI_API_KEY)를 넣어 두므로 객체 구성 단계가
     키 부재로 막히지 않습니다. 실제 모델 호출은 하지 않습니다.

2) 실제 키가 있을 때만 도는 '실제 호출 테스트'
   - @pytest.mark.skipif로, 진짜 키가 없으면 건너뜁니다(skip).
   - 키가 있으면 최소 호출 1~2건을 실제로 실행해, 도구가 묶인 모델 호출이
     tool_calls를 돌려주고 tool_choice='none'이 도구를 막는지 확인합니다.

실제 호출 테스트는 강의 직전에 강사가 키를 넣고 1회 돌려 확인하는 용도입니다.
"""
import importlib.util
import os
import py_compile
from pathlib import Path

import pytest

# 이 테스트 파일(tests/)의 부모가 레포 루트(ai-agent-dev-lgens)입니다.
REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIR = REPO_ROOT / "03_tool_calling"

# 챕터의 6개 예제 파일. 번호 순서대로 둡니다.
EXAMPLE_FILES = [
    "01_define_and_bind.py",
    "02_tool_calls_anatomy.py",
    "03_manual_loop.py",
    "04_parallel_calls.py",
    "05_tool_error.py",
    "06_tool_choice.py",
]

# conftest.py가 setdefault로 넣는 더미 키 값. 이 값이면 '진짜 키 없음'으로 봅니다.
_DUMMY_OPENAI_KEY = "sk-test-dummy-not-used"


def _has_real_openai_key() -> bool:
    """conftest의 더미 키가 아니라 진짜 OPENAI_API_KEY가 있는지 판별합니다."""
    key = os.getenv("OPENAI_API_KEY")
    return bool(key) and key != _DUMMY_OPENAI_KEY


def _load_module(filename: str):
    """파일 경로로 모듈을 직접 로드합니다(숫자로 시작하는 파일명은 일반 import가 안 됨).

    파일 상단의 import·load_dotenv·MODEL 상수·도구 정의까지 실제로 실행되지만,
    main()은 `if __name__ == "__main__"` 안에 있어 자동 실행되지 않습니다.
    따라서 모델 호출 없이 모듈 정의만 메모리에 올라옵니다.
    """
    path = CHAPTER_DIR / filename
    # 모듈 이름은 충돌을 피하려 파일명 앞에 접두사를 붙입니다.
    mod_name = "tool_calling_" + filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1) 키 없이 도는 계약 테스트
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_files_exist(filename):
    # 6개 예제 파일이 모두 챕터 폴더에 실제로 존재하는지 확인합니다.
    assert (CHAPTER_DIR / filename).is_file(), f"{filename} 이 보이지 않습니다."


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_files_compile(filename):
    # 각 예제 파일이 문법 오류 없이 컴파일되는지 확인합니다(키 불필요).
    py_compile.compile(str(CHAPTER_DIR / filename), doraise=True)


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_files_import(filename):
    # 각 예제 파일이 import 가능한지(상단 import·MODEL 상수·도구 정의까지 실행) 확인합니다.
    # main()은 자동 실행되지 않으므로 모델 호출은 일어나지 않습니다.
    module = _load_module(filename)
    # 모든 예제는 main 함수와 MODEL 상수를 가집니다.
    assert hasattr(module, "main"), f"{filename} 에 main()이 없습니다."
    assert module.MODEL == "openai:gpt-5.4-mini"


def test_lab_py_removed():
    # 분리 전의 단일 lab.py는 6개 예제로 쪼개졌으므로 더는 존재하지 않아야 합니다.
    assert not (CHAPTER_DIR / "lab.py").exists(), "lab.py 가 아직 남아 있습니다(분리 후 삭제 대상)."


def test_each_example_has_paired_readme():
    # 각 .py는 동일 이름의 .md(설계·원리) 문서와 1:1로 짝을 이룹니다.
    for filename in EXAMPLE_FILES:
        md = CHAPTER_DIR / filename.replace(".py", ".md")
        assert md.is_file(), f"{md.name} (짝 README)가 보이지 않습니다."
        # 각 README에는 구동 흐름을 보여 주는 mermaid 다이어그램 블록이 있어야 합니다.
        text = md.read_text(encoding="utf-8")
        assert "```mermaid" in text, f"{md.name} 에 mermaid 다이어그램이 없습니다."


def test_tool_decorator_extracts_metadata():
    # 01 예제의 핵심: @tool이 함수의 이름·docstring·타입 힌트에서 메타데이터를 뽑는지 확인합니다.
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    # 함수 이름이 도구 이름이 됩니다.
    assert add.name == "add"
    # docstring이 도구 설명이 됩니다.
    assert add.description == "두 정수를 더한다."
    # 타입 힌트가 인자 스키마(JSON 스키마)로 변환됩니다.
    assert set(add.args) == {"a", "b"}
    assert add.args["a"]["type"] == "integer"

    # @tool로 감싼 함수는 모델 없이 직접 실행할 수 있습니다.
    assert add.invoke({"a": 3, "b": 5}) == 8


def test_bind_tools_returns_runnable():
    # 01 예제의 핵심: bind_tools가 invoke 가능한 Runnable을 돌려주는지(실제 호출 없이) 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain_core.runnables import Runnable
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    model = init_chat_model("openai:gpt-5.4-mini")  # 구성만, invoke 안 함
    bound = model.bind_tools([add])
    assert isinstance(bound, Runnable)
    assert hasattr(bound, "invoke")


def test_bind_tools_accepts_parallel_and_tool_choice():
    # 04·06 예제의 핵심: parallel_tool_calls·tool_choice 옵션이 결합되어 Runnable을 돌려주는지 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain_core.runnables import Runnable
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """두 정수를 곱한다."""
        return a * b

    model = init_chat_model("openai:gpt-5.4-mini")  # 구성만, invoke 안 함

    # 병렬 제어 옵션 결합(04)
    parallel_off = model.bind_tools([add], parallel_tool_calls=False)
    assert isinstance(parallel_off, Runnable)

    # tool_choice 강제·금지·지정 결합(06)
    for choice in ("any", "none", "add"):
        bound = model.bind_tools([add, multiply], tool_choice=choice)
        assert isinstance(bound, Runnable)
        assert hasattr(bound, "invoke")


def test_tool_message_pairs_with_tool_call_id():
    # 02 예제의 핵심: ToolMessage가 tool_call_id로 호출과 결과를 짝짓는 구조인지(호출 없이) 확인합니다.
    from langchain.messages import ToolMessage

    tm = ToolMessage(content="8", tool_call_id="call_abc")
    assert tm.content == "8"
    # 이 id가 모델이 보낸 호출 id와 일치해야 결과가 그 호출의 답으로 묶입니다.
    assert tm.tool_call_id == "call_abc"


def test_tool_exception_raisable_and_carries_message():
    # 05 예제의 핵심: 도구가 ToolException을 던지고, 그 메시지를 그대로 담는지 확인합니다.
    from langchain_core.tools import ToolException

    with pytest.raises(ToolException) as excinfo:
        raise ToolException("0으로는 나눌 수 없습니다.")
    # 잡힌 예외 메시지가 모델에게 되돌릴 내용 그대로입니다.
    assert "0으로는 나눌 수 없습니다." in str(excinfo.value)


# --------------------------------------------------------------------------
# 2) 실제 키가 있을 때만 도는 실제 호출 테스트
#    진짜 키가 없으면 skip 됩니다(더미 키로는 호출하지 않습니다).
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_bound_model_returns_tool_calls_live():
    # 도구가 묶인 모델에 계산 질문을 던지면 tool_calls가 채워지는지 실제로 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain.messages import HumanMessage
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    model = init_chat_model("openai:gpt-5.4-mini", temperature=0.0)
    ai = model.bind_tools([add]).invoke([HumanMessage("3 더하기 5는 얼마야?")])

    # 모델은 답을 직접 쓰지 않고 도구 호출을 제안합니다.
    assert ai.tool_calls, "도구 호출 제안(tool_calls)이 비어 있습니다."
    first = ai.tool_calls[0]
    # 한 호출 항목은 name·args·id 세 필드를 갖춥니다.
    assert first["name"] == "add"
    assert set(first["args"]) == {"a", "b"}
    assert first["id"]


@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_tool_choice_none_blocks_tool_call_live():
    # tool_choice='none'이면 계산 질문이어도 도구를 부르지 않고 직접 답하는지 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain.messages import HumanMessage
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """두 정수를 더한다."""
        return a + b

    model = init_chat_model("openai:gpt-5.4-mini", temperature=0.0)
    ai = model.bind_tools([add], tool_choice="none").invoke([HumanMessage("3 더하기 5는?")])

    # 금지 상태에서는 tool_calls가 비고 content에 직접 답이 들어옵니다.
    assert ai.tool_calls == []
    assert isinstance(ai.content, str) and ai.content.strip()
