"""01 - 상태(State)를 정의하고, 노드 1개짜리 최소 그래프를 만든다.

이 예제 하나만으로 다음을 익힙니다.
  - TypedDict로 상태(State)의 칸(키)을 미리 선언한다.
  - StateGraph로 그래프 빌더를 만들고, 노드 하나를 등록한다.
  - START에서 노드로, 노드에서 END로 엣지를 잇는다.
  - compile로 실행 가능한 그래프를 만들고 invoke로 한 번 돌린다.

이 파일은 자기완결입니다. 다른 파일이나 main에 의존하지 않으며, 아래 한 줄로 단독 실행됩니다.
  uv run python 05_langgraph_workflow/01_state_and_graph.py

이 예제는 모델을 부르지 않습니다. 따라서 API 키 없이도 그대로 돌아갑니다.
"""

# import는 "다른 파일(라이브러리)에 있는 기능을 끌어와 이 파일에서 쓰겠다"는 선언입니다.
# from A import B 는 "A라는 모듈에서 B만 골라 가져온다"는 뜻입니다.
# TypedDict는 "어떤 키들이 들어 있는 딕셔너리인지"를 미리 적어 두는 타입입니다.
from typing_extensions import TypedDict

# langgraph.graph는 v1 권장 경로입니다.
#   StateGraph: 상태 스키마를 받아 그래프를 조립하는 빌더(설계도) 클래스
#   START: 입력이 들어오는 진입점을 나타내는 특수 노드
#   END: 워크플로우가 끝나는 종착점을 나타내는 특수 노드
from langgraph.graph import StateGraph, START, END

# MODEL은 다른 예제와 형식을 맞추기 위한 상수(고정값)입니다. 이 예제는 모델을 부르지 않습니다.
# 따옴표로 감싼 값("...")은 문자열(글자 데이터)입니다. 대문자 이름은 "고정값"이라는 관례입니다.
MODEL = "openai:gpt-5.4-mini"


# class는 "여러 값을 묶은 새로운 형(型)"을 정의하는 키워드입니다.
# 아래는 "topic과 result라는 두 칸을 가진 상태"라는 형을 선언합니다.
# 상태(State)는 그래프가 도는 동안 노드들이 함께 보고 갱신하는 공유 데이터 묶음입니다.
class State(TypedDict):
    topic: str   # 입력으로 받을 글감 (str = 문자열)
    result: str  # 노드가 채울 산출물


# def는 함수(여러 줄의 동작을 하나의 이름으로 묶은 것)를 정의하는 키워드입니다.
# 노드(node)는 "현재 상태를 받아, 바뀐 부분만 딕셔너리로 돌려주는" 함수입니다.
# 인자 state는 그래프가 넘겨주는 현재 상태이고, 타입 힌트 : State는 "이 칸들을 가진다"는 표시입니다.
def echo(state: State) -> dict:
    # 노드는 상태 전체를 다시 그릴 필요 없이, 자기가 채울 칸만 dict로 돌려줍니다.
    # state['topic']은 상태의 topic 칸 값을 읽는 것입니다(대괄호로 키를 지정).
    # f"..."는 f-string으로, 문자열 안 { } 자리에 변수 값을 끼워 넣어 줍니다.
    return {"result": f"입력받은 글감: {state['topic']}"}


# 화살표 뒤의 -> None은 "이 함수는 값을 돌려주지 않는다"는 타입 힌트(설명용 표시)입니다.
def main() -> None:
    # 1) 빌더를 만듭니다. StateGraph에 상태 스키마(State)를 넘겨, 어떤 칸을 다룰지 알려 줍니다.
    #    = 기호는 "오른쪽 결과를 왼쪽 이름(builder)에 담아 둔다"는 대입입니다.
    builder = StateGraph(State)

    # 2) 노드를 등록합니다. 첫 인자는 노드 이름(문자열), 둘째 인자는 노드가 될 함수입니다.
    #    .add_node(...)는 builder 객체가 가진 메서드(객체에 딸린 함수)를 호출하는 것입니다.
    builder.add_node("echo", echo)

    # 3) 엣지(edge)로 실행 순서를 잇습니다. 엣지는 "노드에서 노드로의 고정 연결"입니다.
    #    START → echo → END. 일반 노드를 잇듯 START·END도 똑같이 add_edge로 연결합니다.
    builder.add_edge(START, "echo")  # 진입점에서 echo로
    builder.add_edge("echo", END)    # echo에서 종착점으로

    # 4) compile로 설계도를 "실행 가능한 그래프"로 만듭니다. 컴파일하지 않으면 invoke할 수 없습니다.
    #    compile은 끊긴 노드가 없는지 같은 기본 구조 검사도 함께 합니다.
    graph = builder.compile()

    # 5) invoke로 한 번 실행합니다. 입력은 "상태 형태의 딕셔너리"입니다.
    #    중괄호 { } 그 자체는 딕셔너리(key: value 짝의 묶음)입니다.
    initial_state = {"topic": "LangGraph", "result": ""}

    # 그래프에 넣기 전 상태를 먼저 출력해, 노드를 거치며 무엇이 바뀌는지 비교할 수 있게 합니다.
    print("=== START → echo → END 그래프를 한 번 실행합니다 ===")
    print("1) 입력 상태:", initial_state)        # result는 아직 빈 문자열입니다.
    print("2) echo 노드 통과 중 ...")            # 이 한 노드가 result 칸을 채웁니다.

    result = graph.invoke(initial_state)

    # 결과도 상태 딕셔너리입니다. echo 노드가 채운 result 칸을 꺼내 출력합니다.
    print("3) 최종 상태:", result)               # result 칸이 채워진 상태 전체를 봅니다.
    print("   - topic (입력 유지):", result["topic"])   # 입력은 그대로 남아 있습니다.
    print("   - result (노드 산출):", result["result"]) # 예: 입력받은 글감: LangGraph

    # 체크포인트:
    #   - State에 어떤 칸을 둘지부터 정한다는 점을 이해하면 출발 준비가 된 것입니다.
    #   - 모델 없이도 START → echo → END가 돌아 result가 채워지면 그래프 뼈대를 이해한 것입니다.
    #   - 노드가 상태 전체가 아니라 "바뀐 칸(result)"만 돌려준다는 점이 핵심입니다.


# 아래 한 줄은 "이 파일을 직접 실행했을 때만 main()을 부른다"는 파이썬의 표준 관용구입니다.
# (다른 파일이 이 파일을 import할 때는 자동 실행되지 않게 막아 줍니다.)
if __name__ == "__main__":
    main()
