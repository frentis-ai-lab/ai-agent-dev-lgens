# AI Agent 개발 실습 (LG에너지솔루션)

LangChain과 LangGraph로 도구를 쓰고 기억하는 AI Agent를 직접 구현하는 과정의 실습 레포입니다. 로컬 Python 환경에서 [uv](https://docs.astral.sh/uv/)로 설치하고 실행합니다. 각 챕터는 강의 주제와 1:1로 대응하며, **주제마다 독립 실행 예제 파일 하나**로 나뉘어 있어 번호 순서대로 하나씩 따라 실행하면 개념이 점점 쌓입니다.

## 폴더 구조

```
ai-agent-dev-lgens/
├── pyproject.toml          # uv 프로젝트 정의 (의존성)
├── .env.example            # API 키 템플릿 (.env로 복사해 사용)
├── 02_langchain_core/      # 챕터 폴더 — 독립 예제 + 짝 README
│   ├── README.md           #   챕터 개요·학습 경로
│   ├── 01_model_call.py    #   독립 실행 예제 (자기완결)
│   ├── 01_model_call.md    #   그 예제의 설계·구동 원리(다이어그램 포함)
│   └── ...                 #   02~ 예제들
├── 03_tool_calling/        # 〃
├── 04_custom_tool/         # 〃
├── 05_langgraph_workflow/  # 〃
├── 06_langgraph_agent/     # 〃
├── 07_short_memory/        # 〃
├── 08_long_memory/         # 〃
├── streamlit/              # 대표 Streamlit 앱 (01~03) — "화면이 있는" 버전
├── tests/                  # pytest (계약 테스트 + 실제 키 게이트 테스트)
└── archive/notebooks/      # 이전 Colab 노트북 (참고용 보관)
```

각 챕터 폴더는 `NN_topic.py`(독립 실행 코드)와 `NN_topic.md`(그 예제만으로 혼자 학습할 수 있는 설계·구동 원리, Mermaid 다이어그램 포함)가 짝을 이루며, 폴더 `README.md`가 학습 경로를 안내합니다.

## 사전 준비

빠른 시작에 앞서 아래 세 가지를 준비합니다.

1. **uv 설치** — Python 패키지·환경 관리 도구입니다. 아직 없으면 한 줄로 설치합니다. 설치 방법은 [공식 문서](https://docs.astral.sh/uv/getting-started/installation/)를 참고하십시오.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **OpenAI API 키 발급** — [platform.openai.com](https://platform.openai.com/api-keys)에서 키를 발급합니다. 사용량 과금이므로 결제 수단을 등록하고, 실습용으로 소액(예: 5~10달러) 한도를 걸어 두기를 권장합니다.
3. **Python 3.12 이상** — `uv sync`가 필요한 버전을 자동으로 맞춰 주지만, 시스템에 3.12 이상이 있으면 더 매끄럽습니다.

## 빠른 시작

```bash
# 1) 의존성 설치 (uv가 .venv를 만들고 고정 버전을 설치합니다)
uv sync

# 2) API 키 설정 — .env.example을 복사해 키를 채웁니다 (코드에 키를 적지 않습니다)
cp .env.example .env
#   .env 파일에 OPENAI_API_KEY=sk-... 입력 (Gemini 예제를 보려면 GOOGLE_API_KEY도)

# 3) 예제는 하나씩 단독으로 실행합니다 (번호 순서 권장)
uv run python 02_langchain_core/01_model_call.py
uv run python 02_langchain_core/02_messages_context.py
# ... 각 챕터의 README.md가 예제 순서를 안내합니다

# Streamlit 앱 실행
uv run streamlit run streamlit/01_chat.py
```

각 예제 파일은 상단에 `load_dotenv()`·`MODEL` 상수·자체 모델 초기화를 갖춰 다른 파일에 의존하지 않습니다. 키가 없어도 문법·import 점검은 됩니다.

## 실습 챕터

각 챕터 폴더에는 독립 실행 예제(`NN_topic.py`)와 짝 README(`NN_topic.md`)가 들어 있습니다. 챕터 폴더의 `README.md`에서 예제 순서와 학습 경로를 보십시오.

| 챕터 폴더 | 주제 | 예제 | 핵심 내용 |
|-----------|------|------|-----------|
| `02_langchain_core/` | LangChain 핵심 구성요소 | 6 | 모델 호출·메시지·맥락 누적·파라미터·스트리밍·LCEL·구조화 출력 |
| `03_tool_calling/` | 도구 호출 | 6 | 도구 정의·bind_tools·tool_calls 해부·수동 루프(MAX_TURNS)·병렬·에러·tool_choice |
| `04_custom_tool/` | Custom Tool과 시스템 프롬프트 | 6 | args_schema·description 영향·시스템 프롬프트 설계·ToolException·승인 게이트 |
| `05_langgraph_workflow/` | LangGraph 워크플로우 | 6 | 상태·노드·엣지·리듀서·조건 분기·라우터 유형·순환과 recursion_limit |
| `06_langgraph_agent/` | LangGraph 기반 Agent | 6 | 수동 그래프(ToolNode·tools_condition)·ReAct 루프·create_agent·다중 도구·에러·안전 |
| `07_short_memory/` | 단기 메모리 | 6 | 메모리 없는 한계·checkpointer·thread_id 멀티턴·상태 조회·trim·요약·영속 |
| `08_long_memory/` | 장기 메모리와 회상 | 8 | Store·시맨틱 인덱스·네임스페이스·구조형 vs 시맨틱·in-graph vs tool-call·교차 세션 |

> AI Agent 개념·구조(첫 주제)는 이론이라 코드 실습이 없습니다. 실습은 02부터 08까지입니다.

## Streamlit 앱 (`streamlit/`)

같은 로직을 "화면이 있는" 형태로 보여 주는 대표 앱 3종입니다.

| 번호 | 앱 | 파일 | 대응 챕터 | 보여 주는 것 |
|------|----|------|-----------|--------------|
| 01 | 챗봇 | `streamlit/01_chat.py` | 02 | 멀티턴 대화·시스템 프롬프트·스트리밍·구조화 출력 토글 |
| 02 | 도구 Agent | `streamlit/02_tool_agent.py` | 03~06 | 도구를 쓰는 Agent와 도구 호출 과정 가시화 |
| 03 | 기억 Agent | `streamlit/03_memory_agent.py` | 07~08 | thread_id로 단기 맥락, Store로 세션을 넘는 장기 회상 |

```bash
uv run streamlit run streamlit/02_tool_agent.py
```

## 테스트

```bash
uv run pytest
```

각 챕터마다 테스트가 있습니다. **키 없이 도는 계약 테스트**(import 경로·도구 메타데이터·그래프 컴파일·메모리 구성 등 핵심 API의 형태 점검)와, `.env`에 실제 키가 있을 때만 도는 **실제 키 게이트 테스트**(모델 호출·구조화 출력·도구 호출·메모리 회상을 실호출로 검증)로 나뉩니다. 키가 없으면 게이트 테스트는 자동으로 건너뜁니다.

## 모델

- 기본: `openai:gpt-5.4-mini`
- 대안: `google-genai:gemini-3.5-flash` (Gemini 사용 시 `GOOGLE_API_KEY` 필요)

각 파일 상단의 `MODEL` 상수만 바꾸면 전체 코드가 그대로 다른 모델을 사용합니다.

## 참고

- API 키는 코드에 직접 적지 않고 `.env`(또는 셸 환경 변수)로만 주입합니다. `.env`는 `.gitignore`로 커밋에서 제외됩니다.
- 이전에 만든 Colab 노트북은 `archive/notebooks/`에 그대로 보관해 두었습니다.
- 미리 만들어진 Agent는 `langchain.agents`의 `create_agent`를 씁니다(`langgraph.prebuilt`의 `create_react_agent`는 옛 예제에서 보이지만 사용하지 않습니다).
