"""05 - 대화가 길어질 때 trim_messages로 토큰을 통제한다 (자르기).

이 예제 하나만으로 다음을 익힙니다.
  - 대화가 길어지면 왜 토큰이 폭증하는지 이해한다 (매 호출마다 전체를 다시 입력).
  - trim_messages로 토큰 상한에 맞춰 오래된 대화를 잘라 낸다.
  - 시스템 메시지(역할 지시)는 보존하면서 최근 대화를 우선 남기는 전략을 쓴다.

trim_messages는 메시지 리스트를 받아 잘라 주는 "순수 함수"입니다. 모델을 호출하지 않습니다.
그래서 이 예제는 키 없이도 그대로 끝까지 동작합니다 (메시지를 손으로 만들어 자릅니다).

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 07_short_memory/05_trim_messages.py
"""

import os

from dotenv import load_dotenv
# trim_messages는 메시지 리스트를 토큰 상한에 맞춰 잘라 주는 순수 함수입니다.
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
# count_tokens_approximately는 tiktoken 없이도 토큰 수를 어림 계산하는 헬퍼입니다 (데모용, 정확치 아님).
# 운영에서는 token_counter에 모델 객체를 넘겨 정확히 셉니다.
from langchain_core.messages.utils import count_tokens_approximately

load_dotenv()

MODEL = "openai:gpt-5.4-mini"


def make_long_conversation():
    """토큰 관리를 실험하려고 긴 대화를 손으로 흉내 낸다 (모델 호출 없음)."""
    # 메모리에 메시지가 무한정 쌓이면 매 호출 입력 토큰(=비용·지연)이 계속 커집니다.
    # 먼저 자를 대상이 될 긴 대화를 만듭니다 (시스템 1개 + 사용자·답변 5쌍 = 11개).
    messages = [SystemMessage("너는 친절한 한국어 비서다.")]
    for i in range(5):
        messages.append(HumanMessage(f"{i}번째 질문입니다. " * 8))
        messages.append(AIMessage(f"{i}번째 답변입니다. " * 8))
    return messages


def main() -> None:
    # 이 예제는 trim_messages만 쓰므로 키 없이도 끝까지 동작합니다.
    # (다른 예제와의 일관성을 위해 안내만 남깁니다.)
    if not os.getenv("OPENAI_API_KEY"):
        print("[참고] 이 예제는 모델을 호출하지 않아 키 없이도 동작합니다.\n")

    # 1) 자르기 전: 긴 대화를 만들고 지금 토큰 수를 봅니다 (기준값).
    messages = make_long_conversation()
    print("[자르기 전] 메시지", len(messages), "개, 토큰~", count_tokens_approximately(messages))

    # 2) trim_messages로 토큰 상한에 맞춰 잘라 냅니다.
    trimmed = trim_messages(
        messages,
        max_tokens=120,                            # 이 토큰 수 이하로 맞춥니다 (시스템 + 최근 몇 턴만 남음)
        token_counter=count_tokens_approximately,  # 토큰 세는 방법 (운영은 모델 객체를 넘겨 정확히 셉니다)
        strategy="last",                           # 최근 대화를 우선 남깁니다 (오래된 것부터 버림)
        include_system=True,                       # 시스템 메시지(역할 지시)는 항상 보존합니다
        start_on="human",                          # 잘린 첫 메시지는 human부터 시작하도록 정렬 (대화 짝 보존)
    )

    # 3) 자른 후: 토큰이 줄고, 시스템 메시지가 남았는지 확인합니다.
    print("[자른 후]  메시지", len(trimmed), "개, 토큰~", count_tokens_approximately(trimmed))
    print("[남은 종류]", [type(m).__name__ for m in trimmed])  # 예: ['SystemMessage', 'HumanMessage', ...]

    # 실전에서는 이 trim 로직을 에이전트 호출 직전에 끼워 넣어 입력 토큰을 통제합니다.
    # (요약 방식은 다음 예제 06에서 다룹니다. 자르기는 버리고, 요약은 압축해 보존합니다.)

    # 체크포인트:
    #   - 자른 후 토큰 수가 max_tokens(120) 이하로 줄면 성공입니다.
    #   - 남은 종류 맨 앞에 SystemMessage가 그대로 있으면 역할 지시가 보존된 것입니다.
    #   - max_tokens를 줄였다 늘렸다 하며 남는 메시지가 어떻게 달라지는지 관찰해 보십시오.


if __name__ == "__main__":
    main()
