"""02_langchain_core 챕터 테스트 — 계약 테스트 + 실제 키 게이트 테스트.

이 파일은 두 묶음으로 나뉩니다.

1) 키 없이 도는 '계약 테스트' (대부분)
   - 각 예제 파일이 import·py_compile 되는지 확인합니다.
   - 핵심 객체(ChatPromptTemplate 조립, Pydantic 스키마 정의,
     with_structured_output가 Runnable을 반환) 구성을 키 없이 검증합니다.
   - conftest.py가 더미 키(OPENAI_API_KEY)를 넣어 두므로 객체 구성 단계가
     키 부재로 막히지 않습니다. 실제 모델 호출은 하지 않습니다.

2) 실제 키가 있을 때만 도는 '실제 호출 테스트'
   - @pytest.mark.skipif로, 진짜 키가 없으면 건너뜁니다(skip).
   - 키가 있으면 최소 호출 1~2건을 실제로 실행해, 모델 invoke가 AIMessage를
     돌려주고 구조화 출력이 지정 타입 객체를 돌려주는지 확인합니다.

실제 호출 테스트는 강의 직전에 강사가 키를 넣고 1회 돌려 확인하는 용도입니다.
"""
import importlib.util
import os
import py_compile
from pathlib import Path

import pytest

# 이 테스트 파일(tests/)의 부모가 레포 루트(ai-agent-dev-lgens)입니다.
REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIR = REPO_ROOT / "02_langchain_core"

# 챕터의 6개 예제 파일. 번호 순서대로 둡니다.
EXAMPLE_FILES = [
    "01_model_call.py",
    "02_messages_context.py",
    "03_params_streaming.py",
    "04_lcel_chain.py",
    "05_structured_output.py",
    "06_structured_advanced.py",
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
    mod_name = "lc_core_" + filename.replace(".py", "")
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
    # 각 예제 파일이 import 가능한지(상단 import·MODEL 상수까지 실행) 확인합니다.
    # main()은 자동 실행되지 않으므로 모델 호출은 일어나지 않습니다.
    module = _load_module(filename)
    # 모든 예제는 main 함수와 MODEL 상수를 가집니다.
    assert hasattr(module, "main"), f"{filename} 에 main()이 없습니다."
    assert module.MODEL == "openai:gpt-5.4-mini"


def test_chatprompttemplate_assembles_with_variables():
    # 04 예제의 핵심: ChatPromptTemplate이 변수 자리를 갖춰 조립되는지(호출 없이) 확인합니다.
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 {역할}이다. 쉽게 설명한다."),
        ("human", "{질문}"),
    ])
    # 양식이 비워 둔 변수 자리를 정확히 인식하는지 확인합니다.
    assert set(prompt.input_variables) == {"역할", "질문"}

    # 값을 채우면 실제 메시지 리스트로 완성됩니다(이 단계도 모델 호출 없음).
    messages = prompt.invoke({"역할": "교사", "질문": "LCEL이 뭐야?"}).messages
    assert len(messages) == 2
    assert messages[0].type == "system" and "교사" in messages[0].content
    assert messages[1].type == "human" and messages[1].content == "LCEL이 뭐야?"


def test_chatprompttemplate_brace_escape():
    # 04 예제의 함정: 리터럴 중괄호는 {{ }}로 이스케이프해야 KeyError가 안 납니다.
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", '다음 JSON 형식으로만 답한다: {{"answer": "..."}}'),
        ("human", "{질문}"),
    ])
    # 이스케이프된 본문은 변수로 오인되지 않으므로 입력 변수는 질문 하나뿐입니다.
    assert prompt.input_variables == ["질문"]
    rendered = prompt.invoke({"질문": "하늘은 무슨 색이야?"}).messages[0].content
    # 이스케이프가 풀려 본문에 진짜 중괄호 한 쌍이 남아야 합니다.
    assert '{"answer": "..."}' in rendered


def test_pydantic_schema_defines_fields_and_types():
    # 05 예제의 핵심: Pydantic 스키마가 필드·타입·기본값을 갖춰 정의되는지 확인합니다.
    from typing import Optional

    from pydantic import BaseModel, Field

    class Person(BaseModel):
        name: str = Field(description="사람의 이름")
        age: Optional[int] = Field(default=None, description="만 나이, 모르면 비워 둔다")

    # 정보가 없으면 Optional 필드는 None으로 안전하게 비워집니다(지어내지 않음).
    p = Person(name="앤디")
    assert p.name == "앤디"
    assert p.age is None

    # 필드 메타데이터(이름·설명)가 모델에게 전달될 형태로 들어 있는지 확인합니다.
    fields = Person.model_fields
    assert set(fields) == {"name", "age"}
    assert fields["age"].description == "만 나이, 모르면 비워 둔다"


def test_nested_pydantic_schema():
    # 06 예제의 핵심: 중첩 스키마(모델 안 모델·리스트)가 정의되고 채워지는지 확인합니다.
    from typing import List, Optional

    from pydantic import BaseModel, Field

    class Address(BaseModel):
        city: str
        country: str

    class PersonDetail(BaseModel):
        name: str
        skills: List[str] = Field(default_factory=list)
        address: Optional[Address] = Field(default=None)

    # 리스트·중첩 객체를 채워 한 객체로 구성합니다.
    p = PersonDetail(
        name="김철수",
        skills=["파이썬", "자바"],
        address=Address(city="서울", country="대한민국"),
    )
    assert p.skills == ["파이썬", "자바"]
    assert p.address.city == "서울"

    # 정보가 없으면 빈 목록·None으로 안전하게 비워집니다.
    bare = PersonDetail(name="이름만")
    assert bare.skills == []
    assert bare.address is None


def test_with_structured_output_returns_runnable():
    # 05·06 예제의 핵심: with_structured_output이 invoke 가능한 Runnable을 돌려주는지
    # (실제 호출 없이) 확인합니다. conftest의 더미 키로 모델 객체 구성까지만 합니다.
    from langchain.chat_models import init_chat_model
    from langchain_core.runnables import Runnable
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    model = init_chat_model("openai:gpt-5.4-mini")  # 구성만, invoke 안 함
    structured = model.with_structured_output(Person)
    assert isinstance(structured, Runnable)
    assert hasattr(structured, "invoke")

    # include_raw=True 분기도 동일하게 Runnable을 돌려줍니다.
    structured_raw = model.with_structured_output(Person, include_raw=True)
    assert hasattr(structured_raw, "invoke")


def test_chain_pipe_composes_to_runnable():
    # 04 예제의 핵심: prompt | model 파이프 합성이 하나의 Runnable이 되는지 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable

    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 {역할}이다."),
        ("human", "{질문}"),
    ])
    model = init_chat_model("openai:gpt-5.4-mini")  # 구성만, invoke 안 함
    chain = prompt | model
    assert isinstance(chain, Runnable)
    assert hasattr(chain, "invoke")


# --------------------------------------------------------------------------
# 2) 실제 키가 있을 때만 도는 실제 호출 테스트
#    진짜 키가 없으면 skip 됩니다(더미 키로는 호출하지 않습니다).
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_invoke_returns_aimessage_live():
    # 모델 invoke가 AIMessage를 돌려주고 본문·토큰 메타데이터를 갖추는지 실제로 확인합니다.
    from langchain.chat_models import init_chat_model
    from langchain.messages import AIMessage, HumanMessage

    model = init_chat_model("openai:gpt-5.4-mini")
    response = model.invoke([HumanMessage("한 단어로만 답해: 대한민국의 수도는?")])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str) and response.content.strip()
    # 응답은 객체이므로 토큰 사용량 메타데이터를 함께 가집니다.
    assert response.usage_metadata is not None


@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_structured_output_returns_typed_object_live():
    # 구조화 출력이 지정한 Pydantic 타입의 객체를 돌려주고, 정수 필드가 정수로 들어오는지 확인합니다.
    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    model = init_chat_model("openai:gpt-5.4-mini", temperature=0.0)
    person = model.with_structured_output(Person).invoke("홍길동은 30살이다")

    assert isinstance(person, Person)
    assert isinstance(person.age, int)  # 문자열 "30"이 아니라 정수 30
