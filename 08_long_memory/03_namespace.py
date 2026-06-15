"""03 - 네임스페이스로 사용자별 기억 공간을 나눈다 (격리).

이 예제 하나만으로 다음을 익힙니다.
  - namespace의 첫 칸을 사용자 ID로 쓰면, 사용자마다 칸이 나뉜다.
  - 키가 같아도(예: "pref") 칸이 다르면 완전히 별개의 기억이다.
  - 같은 query라도 검색하는 네임스페이스에 따라 그 사용자의 기억만 돌려받는다.

단기 메모리의 thread_id가 "대화를 가르는 열쇠"였다면,
장기 메모리의 namespace는 "지식을 가르는 열쇠"입니다.
사용자 정보가 다른 사용자에게 새어 나가면 안 되는 서비스에서 이 격리는 필수입니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 08_long_memory/03_namespace.py

이 예제는 시맨틱 검색을 쓰므로 OPENAI_API_KEY가 필요합니다.
키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os  # 환경변수(OPENAI_API_KEY) 확인에 씁니다.

from dotenv import load_dotenv  # .env 파일을 환경변수로 올려 줍니다.
from langchain.embeddings import init_embeddings  # "벤더:모델명" → 임베딩 모델 객체
from langgraph.store.memory import InMemoryStore  # 장기 메모리 저장소

load_dotenv()

# 임베딩 모델 이름(벤더:모델명). 시맨틱 검색이 텍스트를 벡터로 바꿀 때 씁니다.
EMBED = "openai:text-embedding-3-small"


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    # 시맨틱 인덱스를 켠 Store를 만듭니다 (의미 기반 검색을 쓰기 위함).
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": init_embeddings(EMBED),
            "fields": ["text"],
        }
    )

    # 네임스페이스의 첫 칸을 사용자 ID로 둡니다. 사용자마다 칸이 나뉘어 기억이 섞이지 않습니다.
    # namespace는 튜플(괄호로 묶은 값 묶음)로 계층을 표현합니다 — 여기서는 (사용자, 주제) 두 칸.
    ns_andy = ("user-andy", "memories")  # 앤디의 칸
    ns_bora = ("user-bora", "memories")  # 보라의 칸

    # 키가 같아도(둘 다 "pref") 칸이 다르면 별개의 기억입니다.
    store.put(ns_andy, "pref", {"text": "앤디는 커피를 즐긴다"})
    store.put(ns_bora, "pref", {"text": "보라는 녹차를 즐긴다"})
    print("두 사용자 칸에 같은 key 'pref'로 서로 다른 기억을 저장했습니다.")
    print("  - andy 칸 namespace =", ns_andy)
    print("  - bora 칸 namespace =", ns_bora)

    # 같은 query라도 검색하는 네임스페이스에 따라 그 사용자의 기억만 돌려받습니다.
    # 아래 두 검색은 query가 똑같지만, 넘긴 칸이 다르므로 결과가 격리됩니다.
    print("\n[같은 query '좋아하는 음료' → 칸마다 다른 결과]")
    for it in store.search(ns_andy, query="좋아하는 음료", limit=1):
        print("  andy 칸:", it.value["text"])  # 커피 (앤디 칸만 회상)
    for it in store.search(ns_bora, query="좋아하는 음료", limit=1):
        print("  bora 칸:", it.value["text"])  # 녹차 (보라 칸만 회상)
    print("  → 같은 key·같은 query인데 결과가 다릅니다. 사용자 격리가 동작합니다.")

    # 같은 사용자 안에서도 주제별로 칸을 더 나눌 수 있습니다.
    #   ("user-andy", "preferences") 에는 취향, ("user-andy", "history") 에는 이력을
    #   두면, 취향만 검색할 때 이력이 딸려 오지 않습니다.
    store.put(("user-andy", "preferences"), "tone", {"text": "앤디는 격식체 답변을 선호한다"})
    store.put(("user-andy", "history"), "h1", {"text": "앤디는 지난주 환불을 문의했다"})
    print("\n[같은 사용자 안에서도 주제 칸을 분리]")
    for it in store.search(("user-andy", "preferences"), query="답변 말투 취향", limit=1):
        print("  preferences 칸:", it.value["text"])  # 격식체 (history의 환불 이력은 끼지 않음)

    # 체크포인트:
    #   - 같은 키('pref')·같은 query인데 앤디 칸과 보라 칸 결과가 다르면, 사용자 격리가 된 것입니다.
    #   - 주제 칸(preferences/history)을 더 나눠 취향만 검색되면, 계층 분류를 이해한 것입니다.


if __name__ == "__main__":
    main()
