"""01 - 장기 메모리 저장소(Store)의 네 연산 (put · get · search · delete).

이 예제 하나만으로 다음을 익힙니다.
  - InMemoryStore를 하나 만든다 (색인 없는 가장 단순한 키-값 저장소).
  - put(namespace, key, value)로 기억 한 건을 저장한다 (value는 dict).
  - get(namespace, key)로 키를 정확히 알 때 한 건을 꺼낸다.
  - search(namespace)로 키를 몰라도 그 칸 전체를 훑는다.
  - 같은 키로 다시 put하면 덮어쓰기(갱신)이고, delete로 지운다(망각).

장기 메모리는 단기 메모리(checkpointer)와 다릅니다. 단기는 한 대화(thread) 안에서만
살아 있지만, Store는 thread와 무관한 별도 저장소라 어느 세션에서 저장하든 함께 봅니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 08_long_memory/01_store_basics.py

이 예제는 API 키가 없어도 동작합니다. 색인(시맨틱 검색)을 켜지 않았으므로
임베딩 호출이 일어나지 않고, 순수 키-값 연산만 사용하기 때문입니다.
"""

# import는 "다른 파일(라이브러리)에 있는 기능을 끌어와 이 파일에서 쓰겠다"는 선언입니다.
# from A import B 는 "A라는 모듈에서 B만 골라 가져온다"는 뜻입니다.
# InMemoryStore는 장기 메모리를 메모리(RAM)에 담는 저장소 클래스입니다.
from langgraph.store.memory import InMemoryStore

# NS는 우리가 직접 만든 변수입니다. 대문자 이름은 "고정값(상수)"이라는 관례입니다.
# namespace(네임스페이스)는 기억을 담는 "서랍의 이름표"입니다. 튜플(괄호로 묶은 값 묶음)로
# 계층을 만듭니다. 여기서는 (사용자, 주제) 두 칸으로, 첫 칸이 사용자 ID입니다.
NS = ("user-123", "memories")


# def는 함수(여러 줄의 동작을 하나의 이름으로 묶은 것)를 정의하는 키워드입니다.
# 화살표 뒤의 -> None은 "이 함수는 값을 돌려주지 않는다"는 타입 힌트(설명용 표시)입니다.
def main() -> None:
    # 1) 가장 단순한 Store를 하나 만듭니다.
    #    index= 설정을 주지 않으면 "키-값 저장소"로만 동작합니다 (의미 기반 검색은 02에서).
    #    = 기호는 "오른쪽 결과를 왼쪽 이름(store)에 담아 둔다"는 대입입니다.
    store = InMemoryStore()
    # 객체.속성 형태로 그 객체의 정보를 꺼냅니다. __class__.__name__은 "이 객체의 클래스 이름"입니다.
    print("만든 Store:", store.__class__.__name__)  # 예: InMemoryStore

    # 2) put(namespace, key, value): 기억 한 건을 저장합니다.
    #    namespace는 "어느 칸에", key는 "어떤 이름표로", value는 "무슨 내용을" 입니다.
    #    value는 반드시 dict(딕셔너리, 중괄호로 묶은 키:값 묶음)입니다.
    store.put(NS, "fact-1", {"text": "앤디는 파이썬을 좋아한다"})
    print("저장 완료: NS =", NS, "/ key = fact-1")

    # 3) get(namespace, key): 키를 정확히 알 때 그 기억 하나를 꺼냅니다 (정확 일치, 가장 빠름).
    #    반환값은 dict가 아니라 항목 객체입니다. 실제 내용은 .value에 들어 있습니다.
    item = store.get(NS, "fact-1")
    print("[get fact-1] .value =", item.value)  # 예: {'text': '앤디는 파이썬을 좋아한다'}

    # 4) 같은 칸(NS)에 기억을 더 쌓고, 키를 몰라도 search로 그 칸 전체를 훑습니다.
    #    color 같은 추가 필드를 함께 넣어 두면, 값을 구조화해 보관할 수 있습니다.
    store.put(NS, "fact-2", {"text": "앤디는 매운 음식을 못 먹는다"})
    store.put(NS, "fact-3", {"text": "앤디는 주말마다 등산을 간다"})
    print("[search 전체]")
    # for는 묶음 안의 항목을 하나씩 꺼내 반복하는 문법입니다. it에 항목이 하나씩 담깁니다.
    for it in store.search(NS):
        print("  -", it.key, it.value)  # 저장한 항목들이 모두 나옵니다

    # 5) 같은 키로 다시 put하면 새 값으로 교체됩니다 ("기억의 갱신").
    #    별도의 수정 연산이 없고, put 하나가 쓰기와 갱신을 겸합니다.
    print("[갱신 전] fact-2 =", store.get(NS, "fact-2").value)
    store.put(NS, "fact-2", {"text": "앤디는 이제 매운 음식도 잘 먹는다"})  # 같은 키 → 교체
    print("[갱신 후] fact-2 =", store.get(NS, "fact-2").value)

    # 6) delete(namespace, key): 키를 삭제합니다 (망각).
    #    이후 같은 키의 get은 None(아무 값도 없음)을 돌려줍니다.
    store.delete(NS, "fact-1")
    print("[delete 후] fact-1 =", store.get(NS, "fact-1"))  # None

    # 체크포인트:
    #   - Store 클래스 이름이 오류 없이 출력되면 저장소 준비가 끝난 것입니다.
    #   - get이 put한 값을 .value로 그대로 돌려주면 키 조회를 이해한 것입니다.
    #   - search가 넣은 항목을 모두 보여 주면 전체 훑기를 이해한 것입니다.
    #   - 같은 키 재-put으로 값이 바뀌면 갱신을, delete 뒤 get이 None이면 망각을 이해한 것입니다.


# 아래 한 줄은 "이 파일을 직접 실행했을 때만 main()을 부른다"는 파이썬의 표준 관용구입니다.
# (다른 파일이 이 파일을 import할 때는 자동 실행되지 않게 막아 줍니다.)
if __name__ == "__main__":
    main()
