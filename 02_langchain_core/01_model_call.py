"""01 - 모델 초기화와 첫 호출 (응답 객체 들여다보기).

이 예제 하나만으로 다음을 익힙니다.
  - init_chat_model로 모델 객체를 만든다 (벤더별 클래스를 직접 import하지 않는다).
  - invoke로 한 번 호출한다.
  - 응답이 문자열이 아니라 "객체(AIMessage)"임을 확인한다.
  - (선택) 모델 문자열만 바꾸면 다른 공급사로 전환된다.

이 파일은 자기완결입니다. 다른 파일이나 main에 의존하지 않으며, 아래 한 줄로 단독 실행됩니다.
  uv run python 02_langchain_core/01_model_call.py

키가 없으면 안내만 출력하고 종료합니다 (문법·import 점검은 키 없이도 됩니다).
"""

# import는 "다른 파일(라이브러리)에 있는 기능을 끌어와 이 파일에서 쓰겠다"는 선언입니다.
# os는 파이썬에 기본 내장된 모듈로, 환경변수(컴퓨터에 저장된 설정값)를 읽을 때 씁니다.
import os

# from A import B 는 "A라는 모듈에서 B만 골라 가져온다"는 뜻입니다.
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# v1 권장 경로입니다. langchain_core.messages에서 가져와도 동일하게 동작합니다.
from langchain.messages import HumanMessage

# .env 파일이 있으면 환경변수로 읽어 들입니다 (로컬 실행 시 키를 .env에 둡니다).
# 함수 이름 뒤의 ( )는 "그 함수를 지금 실행하라"는 표시입니다. 괄호 안은 넘겨주는 값(인자)입니다.
load_dotenv()

# MODEL은 우리가 직접 만든 변수(값을 담아 두는 이름표)입니다. 대문자 이름은 "고정값(상수)"이라는 관례입니다.
# 모델은 "벤더:모델명" 문자열 하나로 지정합니다. 이 줄만 바꾸면 전체 코드가 다른 모델을 씁니다.
# 따옴표로 감싼 값("...")은 문자열(글자 데이터)입니다.
# Gemini로 전환하려면: MODEL = "google-genai:gemini-3.5-flash" (GOOGLE_API_KEY 필요)
MODEL = "openai:gpt-5.4-mini"


# def는 함수(여러 줄의 동작을 하나의 이름으로 묶은 것)를 정의하는 키워드입니다.
# 화살표 뒤의 -> None은 "이 함수는 값을 돌려주지 않는다"는 타입 힌트(설명용 표시)입니다.
def main() -> None:
    # if는 "조건이 참일 때만 안쪽 블록을 실행하라"는 분기문입니다. 들여쓰기(공백)로 블록 범위를 표시합니다.
    # os.getenv("이름")은 환경변수 값을 읽습니다. 값이 없으면 None(아무 값도 없음)을 돌려줍니다.
    # not은 참/거짓을 뒤집습니다. 즉 "키가 없으면(not ...)" 안쪽을 실행합니다.
    # 키가 없으면 호출 단계에서 실패합니다. 미리 안내하고 종료합니다.
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되지 않았습니다. .env에 키를 넣고 다시 실행하십시오.")
        print('  예) OPENAI_API_KEY=sk-...   (또는 export OPENAI_API_KEY="sk-...")')
        return  # return은 함수를 여기서 끝내고 빠져나간다는 뜻입니다.

    # 1) 모델 객체를 만듭니다. 아직 호출은 하지 않습니다.
    #    init_chat_model은 "벤더:모델명" 문자열을 받아 그 공급사에 맞는 모델 객체를 돌려줍니다.
    #    덕분에 ChatOpenAI 같은 벤더별 클래스를 직접 import할 필요가 없습니다.
    #    = 기호는 "오른쪽 결과를 왼쪽 이름(model)에 담아 둔다"는 대입입니다.
    model = init_chat_model(MODEL)
    # 객체.속성 형태로 그 객체가 가진 정보를 꺼냅니다. __class__.__name__은 "이 객체의 클래스(종류) 이름"입니다.
    # print(...)는 괄호 안 값을 화면에 출력합니다. 쉼표로 나누면 여러 값을 띄어쓰기로 이어 찍습니다.
    print("만든 모델 클래스:", model.__class__.__name__)  # 예: ChatOpenAI

    # 2) 첫 호출. invoke의 입력은 "메시지 리스트"입니다. 우선 사용자 메시지 하나만 넣습니다.
    #    .invoke(...)는 model 객체가 가진 메서드(객체에 딸린 함수)를 호출하는 것입니다.
    #    대괄호 [ ]는 리스트(값을 순서대로 담는 묶음)입니다. 여기서는 메시지 한 개만 담았습니다.
    #    HumanMessage("...")는 "사람이 보낸 메시지" 객체를 하나 만드는 것입니다.
    response = model.invoke([HumanMessage("LCEL이 한 문장으로 뭐야?")])

    # 3) 응답은 단순 문자열이 아니라 AIMessage 객체입니다. 안을 하나씩 들여다봅니다.
    #    type(값)은 그 값의 타입(종류)을 알려 줍니다. .__name__으로 이름만 꺼냅니다.
    print("응답 타입:", type(response).__name__)        # 예: AIMessage
    print("본문(content):", response.content)            # 사람이 읽는 답변 텍스트
    print("도구 호출(tool_calls):", response.tool_calls)  # 예: []  (도구를 안 붙였으니 비어 있음. 다음 챕터에서 채워집니다)
    print("토큰 사용량(usage):", response.usage_metadata)  # 입력·출력 토큰 수 (비용 가늠)

    # 체크포인트:
    #   - 모델 클래스 이름이 오류 없이 출력되면 모델 준비가 끝난 것입니다.
    #   - content에 한국어 답변이 들어오면 호출에 성공한 것입니다.
    #   - content 외에 tool_calls·usage_metadata가 함께 있으면 응답이 "구조를 가진 객체"임을 이해한 것입니다.


# (선택) 벤더 전환 — 모델 문자열만 바꾸면 같은 코드가 다른 공급사에서 돕니다.
# GOOGLE_API_KEY가 있을 때만 동작합니다. 호기심이 있으면 아래 주석을 풀고 실행해 보십시오.
def optional_switch_vendor() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY가 없어 건너뜁니다. .env에 키를 넣으면 동작합니다.")
        return
    # 클래스 변경 없이 문자열만 교체했습니다. invoke 사용법은 위와 똑같습니다.
    gemini = init_chat_model("google-genai:gemini-3.5-flash")
    print("[gemini]", gemini.invoke([HumanMessage("LangChain을 한 문장으로 설명해줘.")]).content)


# 아래 한 줄은 "이 파일을 직접 실행했을 때만 main()을 부른다"는 파이썬의 표준 관용구입니다.
# (다른 파일이 이 파일을 import할 때는 자동 실행되지 않게 막아 줍니다.)
if __name__ == "__main__":
    main()
    # optional_switch_vendor()  # GOOGLE_API_KEY가 있으면 주석을 풀어 확인해 보십시오.
