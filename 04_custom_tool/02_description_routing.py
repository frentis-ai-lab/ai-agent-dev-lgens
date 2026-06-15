"""02 - description이 도구 선택(라우팅)을 좌우한다 (좋은 설명 vs 빈약한 설명).

이 예제 하나만으로 다음을 익힙니다.
  - 모델은 함수 본문이 아니라 docstring(description)만 보고 도구를 고른다.
  - 같은 동작이라도 설명이 좋으면 정확히 호출되고, 빈약하면 라우팅이 흔들린다.
  - description은 "무엇을 한다"가 아니라 "언제 쓰는지"를 행동 지시문으로 적는다.

이 파일은 자기완결입니다. 아래 한 줄로 단독 실행됩니다.
  uv run python 04_custom_tool/02_description_routing.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# v1 권장 경로입니다. langchain_core.messages에서 가져와도 동일하게 동작합니다.
from langchain.messages import HumanMessage
# @tool은 langchain_core.tools에 있습니다 (도구 정의의 표준 경로).
from langchain_core.tools import tool

# .env 파일이 있으면 환경변수로 읽어 들입니다 (로컬 실행 시 키를 .env에 둡니다).
load_dotenv()

# 모델은 "벤더:모델명" 문자열 하나로 지정합니다. 이 줄만 바꾸면 전체 코드가 다른 모델을 씁니다.
# google-genai로 전환하려면: MODEL = "google-genai:gemini-3.5-flash" (GOOGLE_API_KEY 필요)
MODEL = "openai:gpt-5.4-mini"


# ===========================================================================
# 두 도구의 '동작'은 똑같습니다. 다른 것은 오직 docstring(description)뿐입니다.
# ===========================================================================

@tool("weather_good")
def weather_good(city: str) -> str:
    """특정 도시의 현재 날씨(기온·강수)를 조회한다.
    예: '서울 날씨 어때?' 처럼 사용자가 도시 날씨를 물을 때 이 도구를 사용한다."""  # 언제 쓰는지까지 명시
    return f"{city}: 맑음, 23도"


@tool("weather_bad")
def weather_bad(city: str) -> str:
    """처리한다."""  # 안티패턴: 무엇을·언제 쓰는지 알 수 없는 모호한 설명
    return f"{city}: 맑음, 23도"


def good_description(model) -> None:
    """좋은 설명을 단 도구는 모델이 의도를 정확히 매칭해 부른다."""
    # bind_tools는 "이 도구들을 쓸 수 있다"고 모델에 알려 주는 것입니다 (아직 실행은 안 함).
    good = model.bind_tools([weather_good])
    # tool_calls는 "이 도구를 이렇게 불러 달라"는 모델의 제안 목록입니다.
    print("[좋은 설명] tool_calls:", good.invoke([HumanMessage("부산 날씨 알려줘")]).tool_calls)
    # 예: [{'name': 'weather_good', 'args': {'city': '부산'}, ...}]  (도구를 정확히 호출)
    # 체크포인트: tool_calls에 weather_good 호출이 담기면, 좋은 설명이 라우팅을 이끈 것입니다.


def bad_description(model) -> None:
    """같은 동작이라도 설명이 빈약하면 라우팅이 흔들린다 (good_description과 비교)."""
    bad = model.bind_tools([weather_bad])
    bad_ai = bad.invoke([HumanMessage("부산 날씨 알려줘")])
    print("[빈약한 설명] tool_calls:", bad_ai.tool_calls)  # 비어 있을 수 있음(라우팅 실패)
    print("[빈약한 설명] content :", bad_ai.content)        # 도구 대신 추측 답변이 나올 수 있음
    # 체크포인트: 좋은 설명은 도구를 부르는데 여기서는 라우팅이 흔들리면,
    #            description이 곧 모델의 "사용 설명서"임을 이해한 것입니다.


def main() -> None:
    # 키가 없으면 호출 단계에서 실패합니다. 미리 안내하고 종료합니다.
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return

    model = init_chat_model(MODEL)  # 강의 직전 최신 모델과 가격을 재확인하십시오.

    print("=== 좋은 설명 → 라우팅 성공 ===")
    good_description(model)
    print("\n=== 빈약한 설명 → 라우팅 흔들림 ===")
    bad_description(model)


if __name__ == "__main__":
    main()
