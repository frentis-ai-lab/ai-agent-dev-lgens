"""04_custom_tool 챕터 테스트 — 계약 테스트 + 실제 키 게이트 테스트.

이 파일은 두 묶음으로 나뉩니다.

1) 키 없이 도는 '계약 테스트' (대부분)
   - 각 예제 파일이 import·py_compile 되는지 확인합니다.
   - 핵심 객체(@tool의 args_schema 필드, field_validator 입구 검증,
     bind_tools가 Runnable을 반환, 승인 게이트의 코드 가드)를 키 없이 검증합니다.
   - conftest.py가 더미 키(OPENAI_API_KEY)를 넣어 두므로 객체 구성 단계가
     키 부재로 막히지 않습니다. 실제 모델 호출은 하지 않습니다.

2) 실제 키가 있을 때만 도는 '실제 호출 테스트'
   - @pytest.mark.skipif로, 진짜 키가 없으면 건너뜁니다(skip).
   - 키가 있으면 최소 호출을 실제로 실행해, 시스템 프롬프트 유무로 도구 호출이
     달라지는지(좋은 설명 도구가 라우팅되는지)를 확인합니다.

실제 호출 테스트는 강의 직전에 강사가 키를 넣고 1회 돌려 확인하는 용도입니다.
"""
import importlib.util
import os
import py_compile
from pathlib import Path

import pytest

# 이 테스트 파일(tests/)의 부모가 레포 루트(ai-agent-dev-lgens)입니다.
REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIR = REPO_ROOT / "04_custom_tool"

# 챕터의 6개 예제 파일. 번호 순서대로 둡니다.
EXAMPLE_FILES = [
    "01_tool_with_schema.py",
    "02_description_routing.py",
    "03_multiple_tools.py",
    "04_system_prompt_design.py",
    "05_tool_exception_recovery.py",
    "06_approval_gate.py",
]

# conftest.py가 setdefault로 넣는 더미 키 값. 이 값이면 '진짜 키 없음'으로 봅니다.
_DUMMY_OPENAI_KEY = "sk-test-dummy-not-used"


def _has_real_openai_key() -> bool:
    """conftest의 더미 키가 아니라 진짜 OPENAI_API_KEY가 있는지 판별합니다."""
    key = os.getenv("OPENAI_API_KEY")
    return bool(key) and key != _DUMMY_OPENAI_KEY


def _load_module(filename: str):
    """파일 경로로 모듈을 직접 로드합니다(숫자로 시작하는 파일명은 일반 import가 안 됨).

    파일 상단의 import·load_dotenv·MODEL 상수·@tool 정의까지 실제로 실행되지만,
    main()은 `if __name__ == "__main__"` 안에 있어 자동 실행되지 않습니다.
    따라서 모델 호출 없이 모듈 정의만 메모리에 올라옵니다.
    """
    path = CHAPTER_DIR / filename
    # 모듈 이름은 충돌을 피하려 파일명 앞에 접두사를 붙입니다.
    mod_name = "custom_tool_" + filename.replace(".py", "")
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
    # 각 예제 파일이 import 가능한지(상단 import·@tool 정의까지 실행) 확인합니다.
    # main()은 자동 실행되지 않으므로 모델 호출은 일어나지 않습니다.
    module = _load_module(filename)
    # 모든 예제는 main 함수를 가집니다.
    assert hasattr(module, "main"), f"{filename} 에 main()이 없습니다."
    # LLM을 호출하는 예제는 MODEL 상수를 가집니다(01은 키 불필요라 MODEL이 없습니다).
    if hasattr(module, "MODEL"):
        assert module.MODEL == "openai:gpt-5.4-mini"


def test_lab_py_removed():
    # 단일 lab.py는 독립 예제 파일들로 대체되었으므로 더 이상 존재하지 않아야 합니다.
    assert not (CHAPTER_DIR / "lab.py").exists(), "lab.py가 아직 남아 있습니다(독립 예제로 대체됨)."


def test_tool_with_schema_args_and_validator():
    # 01 예제의 핵심: @tool에 붙인 Pydantic args_schema가 필드를 노출하고,
    # field_validator가 입구에서 잘못된 입력을 막는지 확인합니다(모델 호출 없음).
    from langchain_core.tools import ToolException

    module = _load_module("01_tool_with_schema.py")
    tool = module.check_inventory
    schema = module.InventoryInput

    # @tool이 노출하는 세 가지(이름·설명·인자)가 모델에게 전달될 형태로 들어 있습니다.
    assert tool.name == "check_inventory"
    assert tool.args_schema is schema
    assert set(tool.args.keys()) == {"sku", "warehouse"}

    # args_schema가 필드·기본값·설명을 갖춰 정의되었는지 확인합니다.
    fields = schema.model_fields
    assert set(fields) == {"sku", "warehouse"}
    assert fields["warehouse"].default == "ICN"           # 기본값이 스키마에 박혀 있음
    assert fields["sku"].description                       # 인자 의미가 모델에게 전달됨

    # field_validator가 입구에서 형식을 강제합니다: BAT-로 시작하지 않으면 차단.
    with pytest.raises(Exception):                         # ValidationError 등
        tool.invoke({"sku": "xyz"})

    # 정상 입력은 정규화(공백 제거·대문자화)를 거쳐 통과합니다.
    out = tool.invoke({"sku": " bat-21700 ", "warehouse": "ICN"})
    assert "BAT-21700" in out and "1,240" in out

    # 형식은 맞지만 데이터가 없으면 ToolException으로 사유를 돌려줍니다.
    with pytest.raises(ToolException):
        tool.invoke({"sku": "BAT-21700", "warehouse": "GWJ"})


def test_field_validator_normalizes_and_blocks():
    # 01 예제의 입구 검증을 스키마 단독으로도 확인합니다.
    module = _load_module("01_tool_with_schema.py")
    InventoryInput = module.InventoryInput

    # 정규화: 앞뒤 공백 제거 + 대문자화.
    ok = InventoryInput(sku=" bat-21700 ")
    assert ok.sku == "BAT-21700"
    assert ok.warehouse == "ICN"                           # 기본값 적용

    # 업무 규칙 위반(BAT- 미시작)은 입구에서 막힙니다.
    with pytest.raises(Exception):
        InventoryInput(sku="CELL-9")


def test_description_routing_tools_differ_only_by_docstring():
    # 02 예제의 핵심: 두 도구의 본문은 같고 description(docstring)만 다른지 확인합니다.
    module = _load_module("02_description_routing.py")
    good = module.weather_good
    bad = module.weather_bad

    # 같은 동작(같은 반환값)이지만 설명은 다릅니다.
    assert good.invoke({"city": "부산"}) == bad.invoke({"city": "부산"})
    assert "사용한다" in good.description                  # 언제 쓰는지 행동 지시문 포함
    assert good.description != bad.description
    assert len(bad.description) < len(good.description)    # 빈약한 설명이 더 짧음


def test_multiple_tools_routing_helpers():
    # 03 예제의 핵심: 서로 다른 두 도구와 수동 루프 헬퍼가 갖춰졌는지(모델 호출 없이) 확인합니다.
    from langchain_core.tools import ToolException

    module = _load_module("03_multiple_tools.py")
    tools = [module.check_inventory, module.convert_currency]
    tool_map = {t.name: t for t in tools}

    assert set(tool_map) == {"check_inventory", "convert_currency"}
    assert callable(module.run_tool_loop)

    # 각 도구는 자기 책임의 입력에 올바른 결과를 냅니다(단일 책임).
    assert "1,240" in module.check_inventory.invoke({"sku": "BAT-21700", "warehouse": "ICN"})
    assert "KRW" in module.convert_currency.invoke({"amount": 100, "currency": "USD"})

    # 지원하지 않는 통화는 ToolException으로 회신합니다.
    with pytest.raises(ToolException):
        module.convert_currency.invoke({"amount": 100, "currency": "XYZ"})


def test_bind_tools_returns_runnable():
    # 02·03·04 예제의 공통: bind_tools가 invoke 가능한 Runnable을 돌려주는지(실호출 없이) 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain_core.runnables import Runnable

    module = _load_module("03_multiple_tools.py")
    model = init_chat_model("openai:gpt-5.4-mini")  # 구성만, invoke 안 함
    bound = model.bind_tools([module.check_inventory, module.convert_currency])
    assert isinstance(bound, Runnable)
    assert hasattr(bound, "invoke")


def test_system_prompt_four_elements_present():
    # 04 예제의 핵심: 좋은 프롬프트가 네 요소(역할·제약·예시·형식)를 갖췄는지 확인합니다.
    # SystemMessage 객체를 직접 만들지 않고, 예제가 쓰는 도구·헬퍼 구성만 점검합니다.
    module = _load_module("04_system_prompt_design.py")
    # 프롬프트 비교 함수들이 모두 정의되어 있어야 합니다.
    for fn in ("good_prompt", "weak_prompt", "antipattern_vague", "antipattern_overload"):
        assert hasattr(module, fn), f"04 예제에 {fn}가 없습니다."
    assert callable(module.run_tool_loop)


def test_approval_gate_code_guard_blocks_without_confirm():
    # 06 예제의 핵심: 승인 게이트가 confirmed 없는 호출을 코드에서 막고,
    # confirmed=True일 때만 실행하는지(모델 호출 없이) 확인합니다.
    from langchain_core.tools import ToolException

    module = _load_module("06_approval_gate.py")
    adjust = module.adjust_inventory

    # confirmed 없는 호출은 코드 가드에 막힙니다(프롬프트와 무관).
    with pytest.raises(ToolException):
        adjust.invoke({"sku": "BAT-21700", "delta": -100})

    # 사람이 확인(confirmed=True)한 뒤에만 실제 실행됩니다.
    out = adjust.invoke({"sku": "BAT-21700", "delta": -100, "confirmed": True})
    assert "-100" in out


def test_tool_exception_recovery_tool_raises_on_missing():
    # 05 예제의 핵심: 없는 데이터에서 도구가 ToolException으로 사유를 돌려주는지 확인합니다.
    from langchain_core.tools import ToolException

    module = _load_module("05_tool_exception_recovery.py")
    tool = module.check_inventory

    assert "1,240" in tool.invoke({"sku": "BAT-21700", "warehouse": "ICN"})
    with pytest.raises(ToolException):
        tool.invoke({"sku": "BAT-21700", "warehouse": "GWJ"})  # 광주 데이터 없음


# --------------------------------------------------------------------------
# 2) 실제 키가 있을 때만 도는 실제 호출 테스트
#    진짜 키가 없으면 skip 됩니다(더미 키로는 호출하지 않습니다).
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_good_description_routes_to_tool_live():
    # 좋은 설명을 단 도구는 같은 질문에서 실제로 tool_calls에 담기는지 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain.messages import HumanMessage

    module = _load_module("02_description_routing.py")
    model = init_chat_model("openai:gpt-5.4-mini", temperature=0.0)
    bound = model.bind_tools([module.weather_good])
    ai = bound.invoke([HumanMessage("부산 날씨 알려줘")])

    # 좋은 설명은 도구 호출 제안을 이끕니다.
    assert ai.tool_calls, "좋은 설명인데 도구가 호출되지 않았습니다."
    assert ai.tool_calls[0]["name"] == "weather_good"


@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_system_prompt_changes_tool_use_live():
    # 시스템 프롬프트 유무로 도구 사용이 달라지는지 최소 검증합니다.
    # '반드시 도구로 확인하라' 제약을 준 프롬프트에서는 도구 호출이 일어나야 합니다.
    from langchain.chat_models import init_chat_model
    from langchain.messages import SystemMessage, HumanMessage

    module = _load_module("04_system_prompt_design.py")
    model = init_chat_model("openai:gpt-5.4-mini", temperature=0.0)
    bound = model.bind_tools([module.check_inventory])

    strict = SystemMessage(
        "너는 사내 재고를 조회하는 물류 비서다. "
        "재고 수량은 절대 추측하지 말고 반드시 check_inventory 도구로 확인하라."
    )
    ai = bound.invoke([strict, HumanMessage("BAT-21700 인천 창고 재고 얼마야?")])
    # 강한 제약 아래에서는 도구로 확인하려는 호출이 담겨야 합니다.
    assert ai.tool_calls, "도구 확인 제약을 줬는데 도구가 호출되지 않았습니다."
    assert ai.tool_calls[0]["name"] == "check_inventory"
