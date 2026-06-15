"""08_long_memory 챕터 테스트 — 계약 테스트 + 실제 키 게이트 테스트.

이 파일은 두 묶음으로 나뉩니다.

1) 키 없이 도는 '계약 테스트' (대부분)
   - 각 예제 파일이 존재·py_compile·import 되고, 짝 README가 함께 있는지 확인합니다.
   - 옛 단일 파일(lab.py)이 삭제되어 파일별 예제 구조로 재편되었는지 확인합니다.
   - 장기 메모리 핵심 API 계약을 키 없이 검증합니다:
       * 색인 없는 InMemoryStore의 put/get/search/delete 연산
       * 시맨틱 인덱스(dims·embed·fields) 설정으로 Store가 구성되는지
       * namespace 격리 — 같은 key라도 다른 칸이면 별개 항목인지
       * langmem 도구(manage_memory·search_memory)가 생성되는지
       * In-graph 노드가 (*, store) 주입을 받고, 방어적 읽기가 섞인 형태를 안전히 처리하는지
   - conftest.py가 더미 키(OPENAI_API_KEY)를 넣어 두므로 임베딩 객체 구성 단계가
     키 부재로 막히지 않습니다. 실제 임베딩·모델 호출은 이 묶음에서 하지 않습니다.

2) 실제 키가 있을 때만 도는 '실제 호출 테스트'
   - @pytest.mark.skipif로, 진짜 키가 없으면 건너뜁니다(skip).
   - 키가 있으면 시맨틱 인덱스를 켠 Store에 put한 뒤 query로 회상해, 단어가 겹치지
     않아도 의미가 가까운 기억이 위로 올라오는지 최소 검증합니다.

실제 호출 테스트는 강의 직전에 강사가 키를 넣고 1회 돌려 확인하는 용도입니다.
"""
import importlib.util
import os
import py_compile
from pathlib import Path

import pytest

# 이 테스트 파일(tests/)의 부모가 레포 루트(ai-agent-dev-lgens)입니다.
REPO_ROOT = Path(__file__).resolve().parent.parent
CHAPTER_DIR = REPO_ROOT / "08_long_memory"

# 챕터의 8개 예제 파일. 번호 순서대로 둡니다.
EXAMPLE_FILES = [
    "01_store_basics.py",
    "02_semantic_index.py",
    "03_namespace.py",
    "04_structured_vs_semantic.py",
    "05_in_graph_recall.py",
    "06_tool_call_memory.py",
    "07_short_plus_long.py",
    "08_cross_session_recall.py",
]

# conftest.py가 setdefault로 넣는 더미 키 값. 이 값이면 '진짜 키 없음'으로 봅니다.
_DUMMY_OPENAI_KEY = "sk-test-dummy-not-used"

# 모든 예제가 공유하는 임베딩 모델 차원 (text-embedding-3-small의 출력 차원).
EMBED = "openai:text-embedding-3-small"
DIMS = 1536


def _has_real_openai_key() -> bool:
    """conftest의 더미 키가 아니라 진짜 OPENAI_API_KEY가 있는지 판별합니다."""
    key = os.getenv("OPENAI_API_KEY")
    return bool(key) and key != _DUMMY_OPENAI_KEY


def _load_module(filename: str):
    """파일 경로로 모듈을 직접 로드합니다(숫자로 시작하는 파일명은 일반 import가 안 됨).

    파일 상단의 import·load_dotenv·MODEL/EMBED 상수까지 실제로 실행되지만,
    main()은 `if __name__ == "__main__"` 안에 있어 자동 실행되지 않습니다.
    따라서 모델·임베딩 호출 없이 모듈 정의만 메모리에 올라옵니다.
    """
    path = CHAPTER_DIR / filename
    mod_name = "long_mem_" + filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# 1) 키 없이 도는 계약 테스트 — 파일·컴파일·import·구조
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_files_exist(filename):
    # 8개 예제 파일이 모두 챕터 폴더에 실제로 존재하는지 확인합니다.
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
    # 각 예제 파일이 import 가능한지(상단 import·MODEL/EMBED 상수까지 실행) 확인합니다.
    # main()은 자동 실행되지 않으므로 모델·임베딩 호출은 일어나지 않습니다.
    module = _load_module(filename)
    assert hasattr(module, "main"), f"{filename} 에 main()이 없습니다."


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_has_readme(filename):
    # 각 예제 파일에 짝이 되는 학습 문서(.md)가 함께 있는지 확인합니다.
    md = filename.replace(".py", ".md")
    assert (CHAPTER_DIR / md).is_file(), f"{md} 짝 문서가 보이지 않습니다."


def test_chapter_readme_exists():
    # 챕터 전체 README가 있는지 확인합니다.
    assert (CHAPTER_DIR / "README.md").is_file(), "챕터 README.md가 보이지 않습니다."


def test_examples_use_correct_model_and_embed():
    # 모델·임베딩을 쓰는 예제들이 지정된 최신 모델 상수를 쓰는지 확인합니다.
    # (01은 색인 없는 키-값만 다뤄 EMBED 상수를 두지 않으므로 제외합니다.)
    for filename in EXAMPLE_FILES:
        module = _load_module(filename)
        if hasattr(module, "MODEL"):
            assert module.MODEL == "openai:gpt-5.4-mini", f"{filename} MODEL 상수가 다릅니다."
        if hasattr(module, "EMBED"):
            assert module.EMBED == EMBED, f"{filename} EMBED 상수가 다릅니다."


def test_no_create_react_agent_usage():
    # Tool-call 방식은 v1 표준 create_agent만 써야 합니다(create_react_agent 금지).
    # 소스 텍스트로 회귀 방지 — 잘못된 import가 다시 들어오면 잡습니다.
    for filename in EXAMPLE_FILES:
        text = (CHAPTER_DIR / filename).read_text(encoding="utf-8")
        assert "create_react_agent" not in text, (
            f"{filename} 에 create_react_agent가 있습니다(create_agent로 교체)."
        )


def test_defensive_recall_read_preserved():
    # In-graph 회상에서 r.value['text'] 직접 접근이 아니라 방어적 읽기를 쓰는지
    # 소스 텍스트로 확인합니다(섞인 저장 형태에서 KeyError 방지).
    for filename in ("05_in_graph_recall.py", "07_short_plus_long.py", "08_cross_session_recall.py"):
        text = (CHAPTER_DIR / filename).read_text(encoding="utf-8")
        # 방어적 읽기 패턴(text → content → str)이 있어야 합니다.
        assert 'value.get("text")' in text and 'value.get("content")' in text, (
            f"{filename} 에 방어적 읽기(text→content→str)가 보이지 않습니다."
        )


# --------------------------------------------------------------------------
# 1) 키 없이 도는 계약 테스트 — 장기 메모리 핵심 API
# --------------------------------------------------------------------------

def test_plain_store_put_get_search_delete():
    # 01 예제의 핵심: 색인 없는 Store의 네 연산이 동작하는지 (키 없이) 확인합니다.
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    ns = ("user-123", "memories")

    # put → get: 저장한 값이 .value로 그대로 돌아옵니다.
    store.put(ns, "fact-1", {"text": "앤디는 파이썬을 좋아한다"})
    assert store.get(ns, "fact-1").value == {"text": "앤디는 파이썬을 좋아한다"}

    # put 여러 건 → search: 넣은 항목이 모두 나옵니다(키 없이 둘러보기).
    store.put(ns, "fact-2", {"text": "앤디는 매운 음식을 못 먹는다"})
    keys = {it.key for it in store.search(ns)}
    assert keys == {"fact-1", "fact-2"}

    # 같은 키 재-put: 덮어쓰기(갱신).
    store.put(ns, "fact-2", {"text": "앤디는 이제 매운 음식도 잘 먹는다"})
    assert store.get(ns, "fact-2").value["text"] == "앤디는 이제 매운 음식도 잘 먹는다"

    # delete: 이후 get은 None.
    store.delete(ns, "fact-1")
    assert store.get(ns, "fact-1") is None


def test_semantic_index_store_constructs():
    # 02 예제의 핵심: 시맨틱 인덱스(dims·embed·fields)로 Store가 구성되는지
    # (실제 임베딩 호출 없이) 확인합니다. conftest의 더미 키로 객체 구성까지만 합니다.
    from langchain.embeddings import init_embeddings
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore(
        index={
            "dims": DIMS,
            "embed": init_embeddings(EMBED),  # 구성만, 임베딩 호출 안 함
            "fields": ["text"],
        }
    )
    # 색인 설정이 오류 없이 받아들여져 Store 객체가 만들어지면 성공입니다.
    assert isinstance(store, InMemoryStore)
    assert hasattr(store, "put") and hasattr(store, "search")


def test_namespace_isolates_same_key():
    # 03 예제의 핵심: 같은 key라도 namespace가 다르면 별개 항목인지 (키 없이) 확인합니다.
    # 색인 없는 Store로 검증해 임베딩 호출을 피합니다.
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    ns_andy = ("user-andy", "memories")
    ns_bora = ("user-bora", "memories")

    store.put(ns_andy, "pref", {"text": "앤디는 커피를 즐긴다"})
    store.put(ns_bora, "pref", {"text": "보라는 녹차를 즐긴다"})

    # 키가 같아도 칸이 다르면 서로의 값이 섞이지 않습니다.
    assert store.get(ns_andy, "pref").value["text"] == "앤디는 커피를 즐긴다"
    assert store.get(ns_bora, "pref").value["text"] == "보라는 녹차를 즐긴다"

    # 한 칸의 search에 다른 칸 항목이 끼지 않습니다.
    andy_keys = {it.key for it in store.search(ns_andy)}
    assert andy_keys == {"pref"}
    andy_texts = {it.value["text"] for it in store.search(ns_andy)}
    assert "보라는 녹차를 즐긴다" not in andy_texts


def test_langmem_tools_create():
    # 06 예제의 핵심: langmem 도구가 생성되고 이름이 manage_memory·search_memory인지
    # (키 없이) 확인합니다.
    from langmem import create_manage_memory_tool, create_search_memory_tool

    ns = ("user-123", "memories")
    manage_tool = create_manage_memory_tool(namespace=ns)
    search_tool = create_search_memory_tool(namespace=ns)

    assert manage_tool.name == "manage_memory"
    assert search_tool.name == "search_memory"
    # 도구는 invoke 가능한 Runnable이어야 합니다.
    assert hasattr(manage_tool, "invoke") and hasattr(search_tool, "invoke")


def test_create_agent_importable():
    # 06 예제의 핵심: Tool-call Agent는 v1 표준 create_agent를 씁니다(create_react_agent 아님).
    from langchain.agents import create_agent

    assert callable(create_agent)


def test_in_graph_store_injection_and_defensive_read():
    # 05 예제의 핵심: 노드가 (*, store) 주입을 받고, 회상 값을 방어적으로 읽어
    # 섞인 저장 형태(text 키 vs content 키)를 KeyError 없이 처리하는지 (모델 없이) 확인합니다.
    from langgraph.graph import StateGraph, START, END
    from langgraph.store.base import BaseStore
    from langgraph.store.memory import InMemoryStore
    from typing import Annotated
    from typing_extensions import TypedDict
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    captured = {}

    def node(state: State, *, store: BaseStore):
        # 색인 없는 store라 query 없는 search로 항목을 가져옵니다(임베딩 호출 회피).
        recalled = store.search(("u", "m"))

        # 예제와 동일한 방어적 읽기: text → content → str(value).
        def _text(value):
            if isinstance(value, dict):
                return value.get("text") or value.get("content") or str(value)
            return str(value)

        captured["texts"] = sorted(_text(r.value) for r in recalled)
        return {"messages": []}

    store = InMemoryStore()
    # 직접 put한 {"text": ...}와 도구가 저장한 형태({"content": ...})를 섞어 둡니다.
    store.put(("u", "m"), "direct", {"text": "직접 저장한 사실"})
    store.put(("u", "m"), "tool", {"content": "도구가 저장한 사실"})

    b = StateGraph(State)
    b.add_node("node", node)
    b.add_edge(START, "node")
    b.add_edge("node", END)
    graph = b.compile(store=store)  # compile(store=...) → 노드 (*, store)로 자동 주입

    graph.invoke({"messages": []})

    # 두 형태 모두 KeyError 없이 텍스트로 읽혀야 합니다.
    assert captured["texts"] == ["도구가 저장한 사실", "직접 저장한 사실"]


# --------------------------------------------------------------------------
# 2) 실제 키가 있을 때만 도는 실제 호출 테스트
#    진짜 키가 없으면 skip 됩니다(더미 키로는 호출하지 않습니다).
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_real_openai_key(),
    reason="실제 OPENAI_API_KEY가 없어 건너뜁니다(더미 키는 호출하지 않음).",
)
def test_semantic_recall_live():
    # 시맨틱 인덱스를 켠 Store에 put한 뒤 query로 회상해, 단어가 겹치지 않아도
    # 의미가 가까운 기억이 위로 올라오는지 실제로 확인합니다(임베딩 호출 1회 이상).
    from langchain.embeddings import init_embeddings
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore(
        index={"dims": DIMS, "embed": init_embeddings(EMBED), "fields": ["text"]}
    )
    ns = ("user-123", "memories")
    store.put(ns, "fact-1", {"text": "앤디는 파이썬을 좋아한다"})
    store.put(ns, "fact-2", {"text": "앤디는 주말마다 등산을 간다"})

    # query에 '파이썬'이라는 단어가 없어도 의미가 가까운 기억이 1위로 와야 합니다.
    hits = store.search(ns, query="좋아하는 프로그래밍 언어", limit=2)
    assert hits, "시맨틱 검색 결과가 비어 있습니다."
    assert hits[0].value["text"] == "앤디는 파이썬을 좋아한다"
    # 각 결과에 유사도 점수가 담깁니다.
    assert hits[0].score is not None
