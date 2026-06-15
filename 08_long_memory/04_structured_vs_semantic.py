"""04 - 구조형 기억 vs 시맨틱 기억 (정확 조회와 근사 회상을 나란히 비교).

이 예제 하나만으로 다음을 익힙니다.
  - 구조형 기억: 분명한 키·필드로 저장하고, 키로 정확히 꺼낸다 (결정적·빠름).
  - 시맨틱 기억: 자유 문장으로 저장하고, 자연어 query로 근사 회상한다 (유연·확률적).
  - 두 방식의 쓰임을 구분해, 상황에 맞게 골라 쓴다.

같은 Store(시맨틱 인덱스를 켠)를 쓰더라도, "키로 정확히 꺼내는가(get)" 와
"자연어로 비슷한 걸 찾는가(search query)" 는 전혀 다른 회상 방식입니다.
  - 구조형: 값이 명확한 정보 (환경설정·프로필·권한) → 정확·결정적 조회·갱신
  - 시맨틱: 흐릿한 맥락 (대화 중 알게 된 취향·일화) → 유연·근사 회상
실무에서는 둘을 함께 씁니다: 확정값은 구조형, 흐릿한 맥락은 시맨틱.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 08_long_memory/04_structured_vs_semantic.py

이 예제는 시맨틱 검색을 쓰므로 OPENAI_API_KEY가 필요합니다.
키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os  # 환경변수(OPENAI_API_KEY) 확인에 씁니다.

from dotenv import load_dotenv  # .env 파일을 환경변수로 올려 줍니다.
from langchain.embeddings import init_embeddings  # "벤더:모델명" → 임베딩 모델 객체
from langgraph.store.memory import InMemoryStore  # 장기 메모리 저장소

load_dotenv()

EMBED = "openai:text-embedding-3-small"


# 함수 인자 store 뒤의 ": InMemoryStore"는 "이 자리에 Store 객체가 온다"는 타입 힌트(설명용)입니다.
# 화살표 뒤 "-> None"은 "값을 돌려주지 않는다"는 표시입니다.
def structured_memory(store: InMemoryStore) -> None:
    """구조형 기억: 분명한 키·필드로 저장하고 키로 정확히 꺼낸다."""
    # 구조형 기억은 의미가 분명한 키와 필드로 값을 정확히 저장하고, 키로 정확히 꺼냅니다.
    #   장점: 결정적이고 빠르며 갱신이 명확합니다 (같은 키 덮어쓰기).
    #   한계: 묻는 사람이 키를 알아야 하고, "비슷한 것"을 찾지는 못합니다.
    profile_ns = ("user-123", "profile")

    # 필드를 나눠 구조적으로 저장합니다 (text 외 필드는 인덱스 대상이 아니라 조회용 메타데이터).
    store.put(profile_ns, "settings", {
        "text": "사용자 환경설정",
        "language": "ko",        # 정확히 이 키로 꺼내 쓸 값
        "tone": "formal",        # 분류·설정처럼 값이 명확한 정보에 적합
        "timezone": "Asia/Seoul",
    })

    # 키를 알므로 get으로 정확 조회합니다 (search query가 아님). .value에 저장한 dict가 들어 있습니다.
    settings = store.get(profile_ns, "settings").value
    print("[구조형] key 'settings'로 정확 조회:")
    print("    language =", settings["language"], "/ tone =", settings["tone"])

    # 설정을 바꿀 때도 같은 키에 덮어써 한 항목만 갱신합니다 (어디를 고칠지 분명).
    # {**settings, "tone": "casual"} 의 **는 "기존 dict의 키:값을 그대로 펼쳐 넣어라"는 표시입니다.
    #   → 나머지 필드는 그대로 두고 tone만 "casual"로 바꾼 새 dict가 만들어집니다.
    store.put(profile_ns, "settings", {**settings, "tone": "casual"})
    print("    tone만 갱신 → tone =", store.get(profile_ns, "settings").value["tone"])

    # 체크포인트: 키로 정확히 같은 값을 꺼내고, 한 필드만 정확히 갱신되면 구조형을 이해한 것입니다.


def semantic_memory(store: InMemoryStore) -> None:
    """시맨틱 기억: 자유 문장으로 저장하고 자연어로 근사 회상한다."""
    # 시맨틱 기억은 자유로운 문장으로 저장하고, 자연어 query로 "의미가 가까운 것"을 찾습니다.
    #   장점: 키를 몰라도 회상할 수 있고, 표현이 달라도 의미로 매칭됩니다.
    #   한계: 점수 기반이라 결과가 확률적이고, 정확한 한 건만 보장하지는 못합니다.
    notes_ns = ("user-123", "notes")

    store.put(notes_ns, "n1", {"text": "앤디는 회의 요약을 짧게 받는 걸 선호한다"})
    store.put(notes_ns, "n2", {"text": "앤디는 알림을 오전에만 받고 싶어 한다"})

    print("[시맨틱] query '요약 길이 취향'으로 근사 회상 (키를 몰라도 됨):")
    for it in store.search(notes_ns, query="요약 길이 취향", limit=1):
        # query에 '요약'만 겹치고 '짧게'는 단어가 다른데도 n1이 회상되면 의미 기반 매칭입니다.
        print("    유사도", round(it.score, 3), "|", it.value["text"])

    # 체크포인트: 단어가 겹치지 않아도 의미로 회상되면 시맨틱 기억을 이해한 것입니다.


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    # 한 Store에 구조형과 시맨틱을 함께 둡니다 (네임스페이스만 다르게).
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": init_embeddings(EMBED),
            "fields": ["text"],
        }
    )

    print("=== 구조형 기억 (키로 정확 조회·갱신) ===")
    structured_memory(store)

    print("\n=== 시맨틱 기억 (자연어로 근사 회상) ===")
    semantic_memory(store)

    print("\n=== 정리 ===")
    print("구조형: 정해진 키/필드 → 정확·결정적 조회·갱신 (환경설정·프로필·권한)")
    print("시맨틱: 자유 문장 + 자연어 검색 → 유연·근사 회상 (대화 중 알게 된 취향·일화)")
    print("실무: 확정값은 구조형, 흐릿한 맥락은 시맨틱으로 함께 씀")


if __name__ == "__main__":
    main()
