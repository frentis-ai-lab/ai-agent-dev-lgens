"""05 - 커스텀 상태와 상태 기반 동적 시스템 프롬프트로 Agent를 개인화한다.

이 예제 하나만으로 다음을 익힙니다.
  - create_agent의 기본 상태(AgentState)를 상속해 우리가 쓸 필드(user_name, tier)를 더한다.
  - state_schema로 기본 상태 대신 확장한 상태를 쓰게 한다.
  - @dynamic_prompt 미들웨어로, 매 호출의 상태 값을 읽어 시스템 프롬프트를 그때그때 만든다.

03·04의 system_prompt는 고정 문자열이라 누가 물어도 같은 지시였습니다. 여기서는 상태에 담긴
고객 이름·등급을 읽어, 같은 질문에도 응대 톤을 바꾸는 '개인화'를 봅니다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 06_langgraph_agent/05_custom_state.py

키가 없으면 안내만 출력하고 종료합니다.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
# AgentState는 create_agent의 기본 상태입니다. 이미 messages 칸을 가지고 있어, 필요한 필드만 얹으면 됩니다.
from langchain.agents import AgentState
# @dynamic_prompt는 매 호출마다 상태를 보고 시스템 프롬프트를 동적으로 만드는 미들웨어 데코레이터입니다.
# ModelRequest는 그 시점의 상태·설정을 담아 전달되는 요청 객체입니다.
from langchain.agents.middleware import dynamic_prompt, ModelRequest

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


# create_agent의 기본 상태(AgentState)를 상속해 우리가 쓸 필드를 추가합니다.
# AgentState에는 이미 messages가 들어 있으므로, 상속하면 messages는 그대로 두고 필요한 필드만 얹게 됩니다.
# (클래스 본문에 "이름: 타입"만 적는 것은 타입 힌트로 상태의 칸을 선언하는 방식입니다 — 값 대입이 아닙니다.)
class SupportState(AgentState):
    user_name: str   # 호출할 때 함께 넘기면 프롬프트·도구에서 읽어 쓸 수 있습니다
    tier: str        # 고객 등급(예: 'VIP', 'general') — 응대 톤을 바꾸는 데 씁니다


# @dynamic_prompt: 매 호출마다 현재 상태를 보고 시스템 프롬프트를 동적으로 만듭니다.
# 고정 문자열 system_prompt와 달리, 상태 값(user_name, tier)을 끼워 넣어 개인화할 수 있습니다.
@dynamic_prompt
def support_prompt(request: ModelRequest) -> str:
    # request.state는 그 호출 시점의 상태 딕셔너리입니다. 우리가 넘긴 user_name·tier가 여기 들어옵니다.
    # .get(키, 기본값)은 키가 없을 때 기본값을 돌려줘, 값이 빠져도 안전하게 동작합니다.
    name = request.state.get("user_name", "고객")
    tier = request.state.get("tier", "general")
    tone = "최우선으로 정중하게" if tier == "VIP" else "친절하게"
    # 상태 값을 끼워 만든 이 문자열이 이번 호출의 시스템 프롬프트가 됩니다.
    return f"너는 고객지원 비서다. '{name}'님을 {tone} 응대하라. 모르면 모른다고 답하라."


def build_agent():
    """커스텀 상태(SupportState)와 상태 기반 동적 프롬프트를 가진 Agent를 만든다.

    state_schema로 기본 상태 대신 우리가 확장한 상태를 쓰고, middleware로 동적 프롬프트를 끼웁니다.
    이 예제의 초점은 상태·프롬프트라, 도구는 비워 둡니다(tools=[]).
    """
    agent = create_agent(
        MODEL,
        tools=[],                       # 이 스텝의 초점은 상태·프롬프트라 도구는 비웁니다
        state_schema=SupportState,      # 기본 상태 대신 우리가 확장한 상태를 사용
        middleware=[support_prompt],    # 동적 프롬프트를 미들웨어로 끼웁니다
    )
    return agent


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        return

    agent = build_agent()

    # 무엇을: 질문(messages)은 똑같이 두고 상태(user_name, tier)만 바꿔, 응대가 어떻게 달라지는지 대비합니다.
    # 같은 질문 + 다른 상태 → 다른 응대가 나오면, 상태가 동적 프롬프트를 거쳐 행동에 반영된 것입니다.
    QUESTION = "환불 절차가 궁금해요"
    print("무엇을: 같은 질문에 상태(고객 이름·등급)만 바꿔, 동적 프롬프트가 응대 톤을 바꾸는지 비교합니다.")
    print(f'공통 질문: "{QUESTION}"\n')

    # invoke 입력에 messages뿐 아니라 커스텀 필드(user_name, tier)도 함께 넘깁니다.
    # 같은 질문이라도 tier에 따라 동적 프롬프트가 달라져 응대 톤이 바뀝니다.
    print("=== 상태 A: VIP 고객 (동적 프롬프트가 '최우선으로 정중하게') ===")
    print("  넘긴 상태: user_name='김에너지', tier='VIP'")
    vip = agent.invoke(
        {"messages": QUESTION, "user_name": "김에너지", "tier": "VIP"}
    )
    print("  응대:", vip["messages"][-1].content)

    print("\n=== 상태 B: 일반 고객 (동적 프롬프트가 '친절하게') ===")
    print("  넘긴 상태: user_name='이배터리', tier='general'")
    general = agent.invoke(
        {"messages": QUESTION, "user_name": "이배터리", "tier": "general"}
    )
    print("  응대:", general["messages"][-1].content)

    print("\n출력 요약: 질문은 한 글자도 안 바꿨지만, 상태(이름·등급)가 다르니 호칭·톤이 달라졌습니다.")
    print("        상태가 @dynamic_prompt를 거쳐 그 호출의 시스템 프롬프트로 들어간 결과입니다.")

    # 체크포인트:
    #   - 같은 질문인데 tier에 따라 응대 톤·호칭이 달라지면, 상태가 프롬프트에 반영된 것입니다.
    #   - user_name이 답변에 나타나면, 커스텀 상태 값이 동적 프롬프트로 전달된 것입니다.
    #   - 고정 system_prompt(03·04)와 달리, 호출마다 프롬프트가 새로 만들어진다는 점이 차이입니다.


if __name__ == "__main__":
    main()
