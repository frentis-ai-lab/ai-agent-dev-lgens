"""02 - 시맨틱 인덱스로 '의미가 가까운' 기억을 회상한다 (키워드 일치가 아님).

이 예제 하나만으로 다음을 익힙니다.
  - index={"dims", "embed", "fields"}로 시맨틱 인덱스를 켠 Store를 만든다.
  - put할 때 지정한 필드(text)의 글이 자동으로 벡터로 변환된다.
  - search에 query를 주면 "의미가 가까운 순서"로 돌려준다 (단어가 안 겹쳐도).
  - 결과의 score(유사도)와 limit(상위 N개)를 들여다본다.

01의 색인 없는 Store는 search가 항목을 그대로 돌려줄 뿐, 의미로 정렬하지 못합니다.
여기서 index=로 임베딩 색인을 켜야 "단어가 안 겹쳐도 의미로 회상"이 동작합니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 08_long_memory/02_semantic_index.py

이 예제는 임베딩 호출을 사용하므로 OPENAI_API_KEY가 필요합니다.
키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
# init_embeddings는 "벤더:모델명" 문자열로 임베딩 모델 객체를 만드는 v1 표준 함수입니다.
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

# .env 파일이 있으면 환경변수로 읽어 들입니다 (로컬 실행 시 키를 .env에 둡니다).
load_dotenv()

# 임베딩 모델. 시맨틱 인덱스가 텍스트를 벡터(숫자 배열)로 바꿀 때 씁니다.
# dims(차원)는 이 모델의 출력 차원에 맞춰 1536으로 둡니다 (둘은 한 쌍으로 맞춰야 합니다).
EMBED = "openai:text-embedding-3-small"

# 기억을 담는 네임스페이스. (사용자, 주제) 두 칸입니다.
NS = ("user-123", "memories")


def main() -> None:
    # os.getenv("이름")은 환경변수 값을 읽습니다. 값이 없으면 None을 돌려줍니다.
    # not은 참/거짓을 뒤집습니다. 즉 "키가 없으면(not ...)" 안내 후 종료합니다.
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return  # return은 함수를 여기서 끝내고 빠져나간다는 뜻입니다.

    # 1) 시맨틱 인덱스를 켠 Store를 만듭니다.
    #    index 설정은 세 칸으로 이뤄집니다.
    #      - dims:   임베딩 벡터의 차원 수 (embed 모델의 출력 차원과 같아야 함)
    #      - embed:  텍스트를 벡터로 바꿀 임베딩 모델 객체
    #      - fields: 임베딩할 필드 이름 목록 (value의 text만 벡터화, 나머지 메타데이터는 제외)
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": init_embeddings(EMBED),
            "fields": ["text"],
        }
    )
    print("시맨틱 인덱스 켠 Store 준비 완료 (dims=1536, fields=['text'])")

    # 2) 기억을 몇 건 넣습니다. put 시점에 text 필드가 자동으로 벡터로 변환됩니다.
    store.put(NS, "fact-1", {"text": "앤디는 파이썬을 좋아한다"})
    store.put(NS, "fact-2", {"text": "앤디는 매운 음식을 못 먹는다"})
    store.put(NS, "fact-3", {"text": "앤디는 주말마다 등산을 간다"})

    # 3) query에 '파이썬'이라는 단어가 직접 없어도, 의미가 가까운 기억이 위로 올라옵니다.
    #    limit는 상위 몇 개를 받을지 정합니다. 여기서는 1개만 받습니다.
    print("[query] 좋아하는 프로그래밍 언어")
    for it in store.search(NS, query="좋아하는 프로그래밍 언어", limit=1):
        print("  -", it.value["text"])  # 예: 앤디는 파이썬을 좋아한다

    # 4) search 결과의 각 항목에는 it.score(유사도)가 담깁니다. 높을수록 query에 가깝습니다.
    #    상위 2개를 점수와 함께 받아, 가장 관련 있는 기억이 맨 위에 오는지 봅니다.
    #    round(값, 3)은 소수점 셋째 자리까지 반올림해 읽기 좋게 만듭니다.
    print("[query] 주말 취미 활동 (상위 2개, 점수 포함)")
    for it in store.search(NS, query="주말 취미 활동", limit=2):
        print("  ", round(it.score, 3), it.value["text"])  # 등산 기억이 가장 높은 점수로

    # 변형 포인트: limit를 키우면 덜 관련된 기억까지 끌려와 노이즈가 늘고,
    #            줄이면 회상이 빈약해집니다. 회상 개수는 품질과 토큰의 트레이드오프입니다.

    # 체크포인트:
    #   - 인덱스 설정이 오류 없이 만들어지면 의미 기반 회상을 위한 Store가 준비된 것입니다.
    #   - '프로그래밍 언어' 검색에서 파이썬 기억이 회상되면 시맨틱 검색을 이해한 것입니다.
    #   - 가장 관련 있는 기억이 가장 높은 score로 맨 위에 오면 점수·정렬을 이해한 것입니다.


if __name__ == "__main__":
    main()
