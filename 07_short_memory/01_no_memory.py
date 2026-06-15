"""01 - 메모리 없는 Agent는 매번 처음 만난 것처럼 답한다.

이 예제 하나만으로 다음을 익힙니다.
  - create_agent로 에이전트를 하나 만든다 (checkpointer를 일부러 붙이지 않는다).
  - 같은 질문을 두 번 나눠 호출하면, 두 번째 호출이 첫 대화를 전혀 기억하지 못함을 확인한다.
  - 그 까닭이 "모델이 멍청해서"가 아니라 "호출 사이에 상태를 저장하는 부품이 없어서"임을 이해한다.

이 파일은 자기완결입니다. 다른 파일이나 main에 의존하지 않으며, 아래 한 줄로 단독 실행됩니다.
  uv run python 07_short_memory/01_no_memory.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

# import는 "다른 파일(라이브러리)에 있는 기능을 끌어와 이 파일에서 쓰겠다"는 선언입니다.
# os는 파이썬에 기본 내장된 모듈로, 환경변수(컴퓨터에 저장된 설정값)를 읽을 때 씁니다.
import os

# from A import B 는 "A라는 모듈에서 B만 골라 가져온다"는 뜻입니다.
from dotenv import load_dotenv
# create_agent는 v1에서 권장하는 "한 줄 에이전트" 생성 함수입니다 (모델·도구·메모리를 한 번에 묶어 줍니다).
from langchain.agents import create_agent
# @tool 데코레이터는 평범한 파이썬 함수를 "에이전트가 부를 수 있는 도구"로 바꿔 줍니다.
from langchain_core.tools import tool

# .env 파일이 있으면 환경변수로 읽어 들입니다 (로컬 실행 시 키를 .env에 둡니다).
load_dotenv()

# MODEL은 "벤더:모델명" 문자열 하나로 지정합니다. 이 줄만 바꾸면 전체 코드가 다른 모델을 씁니다.
# google-genai로 전환하려면: MODEL = "google-genai:gemini-3.5-flash" (GOOGLE_API_KEY 필요)
MODEL = "openai:gpt-5.4-mini"


# @tool 한 줄을 함수 위에 붙이면, 이 함수는 에이전트가 호출할 수 있는 도구가 됩니다.
# 함수의 독스트링("""...""")은 모델이 "이 도구가 무엇을 하는지" 읽는 설명입니다.
@tool
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


def main() -> None:
    # os.getenv("이름")은 환경변수 값을 읽습니다. 값이 없으면 None(아무 값도 없음)을 돌려줍니다.
    # not은 참/거짓을 뒤집습니다. 즉 "키가 없으면(not ...)" 안쪽을 실행합니다.
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return  # return은 함수를 여기서 끝내고 빠져나간다는 뜻입니다.

    # 1) 에이전트를 하나 만듭니다. create_agent는 모델명 문자열을 받아 내부에서 모델을 초기화합니다.
    #    여기서는 checkpointer(대화 상태를 저장하는 부품)를 일부러 붙이지 않습니다.
    #    "메모리가 없는 상태"를 먼저 눈으로 보기 위해서입니다.
    agent = create_agent(
        MODEL,
        tools=[add],
        system_prompt="너는 친절한 한국어 비서다.",
        # checkpointer=...  (다음 예제에서 이 한 줄을 더합니다. 지금은 비워 둡니다.)
    )
    # 객체.__class__.__name__은 "이 객체의 클래스(종류) 이름"입니다.
    print("[에이전트]", type(agent).__name__, "(checkpointer 없음)")

    # 2) 첫 번째 호출: 이름을 알려 줍니다.
    #    에이전트 입력은 {"messages": [...]} 형태입니다. 메시지는 {"role": ..., "content": ...} 딕셔너리로 넣습니다.
    print("\n[1번째 호출] 보냄  :", "내 이름은 앤디야. 기억해 줘.")
    r1 = agent.invoke({"messages": [{"role": "user", "content": "내 이름은 앤디야. 기억해 줘."}]})
    # 응답 딕셔너리의 "messages"는 누적된 메시지 리스트입니다. [-1]은 "맨 마지막", 즉 모델의 최신 답변입니다.
    print("[1번째 호출] 답변  :", r1["messages"][-1].content)
    # 이 호출이 끝나면 r1["messages"]는 사라집니다. 다음 호출로 이어지지 않습니다(저장하는 부품이 없으므로).
    print("[1번째 호출] 메시지 수:", len(r1["messages"]))

    # 3) 두 번째 호출: 이름을 물어봅니다.
    #    호출 사이에 상태를 저장하는 부품(checkpointer)이 없으므로,
    #    이 호출은 1번째 대화를 전혀 모르는 "완전히 새로운 대화"입니다.
    print("\n[2번째 호출] 보냄  :", "내 이름이 뭐였지?")
    r2 = agent.invoke({"messages": [{"role": "user", "content": "내 이름이 뭐였지?"}]})
    print("[2번째 호출] 답변  :", r2["messages"][-1].content)  # 이름을 모른다고 답하면 정상입니다.
    # 1턴이 이어졌다면 메시지가 4개(user·ai 두 쌍) 쌓였겠지만, 여기서는 2번째 호출의 user·ai만 들어 2개뿐입니다.
    print("[2번째 호출] 메시지 수:", len(r2["messages"]), "(1번째 대화가 이어지지 않아 누적되지 않음)")

    # 두 호출이 서로를 모른다는 사실을 한눈에 대비합니다.
    print("\n[비교] 두 호출이 각각 독립적이라 1번째에서 알려 준 이름이 2번째로 이어지지 않습니다.")

    # 체크포인트:
    #   - 2번째 호출에서 모델이 이름을 모른다고 답하면, 단기 메모리가 왜 필요한지 체감한 것입니다.
    #   - 모델은 호출 사이에 아무것도 기억하지 못합니다. 매 호출은 독립적입니다(무상태).
    #   - 다음 예제(02)에서 checkpointer 한 줄로 이 문제를 해결합니다.


# 아래 한 줄은 "이 파일을 직접 실행했을 때만 main()을 부른다"는 파이썬의 표준 관용구입니다.
# (다른 파일이 이 파일을 import할 때는 자동 실행되지 않게 막아 줍니다 — 테스트가 이 점을 이용합니다.)
if __name__ == "__main__":
    main()
