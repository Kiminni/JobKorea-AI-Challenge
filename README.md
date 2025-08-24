# JobKorea AI Challenge

AI 기반 이력서 분석 및 맞춤형 학습 경로 생성 서비스

## 주요 기능

- **이력서 분석**: AI를 활용한 이력서 텍스트 분석 및 구조화
- **면접 질문 생성**: 맞춤형 기술 면접 질문 5개 생성 (초급/중급/고급)
- **학습 경로 제안**: 개인 맞춤형 학습 계획 및 실천 방안 3개 생성
- **비동기 처리**: Celery를 통한 대용량 이력서 처리 (Redis Message Queue)
- **실시간 모니터링**: 작업 진행 상황 실시간 추적
- **API 메트릭스**: 상세한 API 사용 통계 및 성능 모니터링

## 아키텍처
<img width="820" height="400" alt="image" src="https://github.com/user-attachments/assets/d54ffd91-0495-4f1e-a4fc-1114afee95fa" />

### 비동기 처리 흐름

1. **FastAPI** → **Redis Message Queue** (작업 요청)
2. **Celery Worker** → **Redis Message Queue** (작업 가져오기)
3. **Worker 처리 완료** → **Redis Result Backend** (결과 저장)
4. **API** → **Redis Result Backend** (결과 조회)

##  기술 스택

- **Backend**: FastAPI, SQLAlchemy
- **AI/ML**: Google Gemini AI (gemini-2.5-flash)
- **비동기 처리**: Celery (Redis 브로커)
- **Message Queue**: Redis (Celery 브로커 + Result Backend)
- **데이터베이스**: PostgreSQL
- **캐싱**: Redis (세션, 캐시)
- **웹 서버**: Nginx
- **컨테이너**: Docker, Docker Compose
- **테스트**: pytest, pytest-asyncio

## 요구사항

- Python 3.10+
- Docker & Docker Compose
- Google Gemini API 키

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd JobKorea-AI-Challenge

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# Google Gemini API 키 설정
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

### 3. 서비스 실행

```bash
# 모든 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 특정 서비스 로그 확인
docker-compose logs -f fastapi_app
docker-compose logs -f celery_worker
```

### 4. 서비스 상태 확인

Swagger UI: http://localhost:8000/docs

```bash
# API 서버 상태 확인
curl http://localhost:8000/health

# Celery 워커 상태 확인
curl http://localhost:8000/api/v1/resume/workers/status

# Celery Flower 모니터링 (웹 브라우저)
open http://localhost:5555
```

## 🔧 API 사용법

### 동기 이력서 분석

```bash
curl -X POST "http://localhost:8000/api/v1/resume/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "백엔드 개발자 3년차, Java, Spring Boot 경험..."
  }'
```

### 비동기 이력서 분석

```bash
# 1. 분석 작업 요청
curl -X POST "http://localhost:8000/api/v1/resume/analyze-async" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "백엔드 개발자 3년차, Java, Spring Boot 경험..."
  }'

# 응답: {"job_id": "task-uuid", "status": "PENDING"}

# 2. 작업 상태 확인
curl "http://localhost:8000/api/v1/resume/job-status/{job_id}"
```

## 🔍 문제 해결

### Celery 워커 문제

```bash
# 워커 상태 확인
docker-compose exec celery_worker celery -A app.workers.celery_app inspect active

# 워커 재시작
docker-compose restart celery_worker

# 로그 확인
docker-compose logs celery_worker
```

### Redis 연결 문제

```bash
# Redis 상태 확인
docker-compose exec redis redis-cli ping

# Redis 재시작
docker-compose restart redis
```

### 데이터베이스 연결 문제

```bash
# PostgreSQL 상태 확인
docker-compose exec postgres pg_isready -U jobkorea -d jobkorea_ai

# 데이터베이스 재시작
docker-compose restart postgres
```

## 📊 모니터링

### API 메트릭스

- **세션 추적**: X-Session-ID 헤더를 통한 사용자 세션 관리
- **성능 모니터링**: 응답 시간, AI 처리 시간 측정
- **오류 추적**: 상세한 오류 로깅 및 분류

### 헬스 체크

```bash
# 헬스 체크
curl http://localhost:8000/health

# 응답 예시:
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "celery": {
    "broker_connected": true,
    "active_workers": 1,
    "registered_tasks": 2
  },
  "services": {
    "database": "connected",
    "redis": "connected",
    "celery_worker": "running"
  }
}
```

## 보안

- API 키는 환경 변수로 관리
- 세션 ID를 통한 요청 추적
- 클라이언트 IP 주소 로깅
- 상세한 오류 로깅으로 보안 모니터링

## 📝 개발 가이드

### 새로운 API 엔드포인트 추가

```python
# app/api/resume.py
@router.post("/new-endpoint")
async def new_endpoint(request: Request, db: AsyncSession = Depends(get_db)):
    session_id = get_session_id(request)
    client_ip = get_client_ip(request)
    
    # 로직 구현
    
    # 메트릭스 로깅
    background_tasks.add_task(
        log_api_metrics,
        db, "/new-endpoint", "POST", session_id, client_ip,
        response_time, status_code
    )
```

### 새로운 Celery 태스크 추가

```python
# app/workers/celery_app.py
@celery_app.task(bind=True, name="app.workers.celery_app.new_task")
def new_task(self, *args, **kwargs):
    try:
        # 작업 로직
        result = do_something(*args, **kwargs)
        
        # 진행률 업데이트
        self.update_state(
            state="PROGRESS",
            meta={"progress": 100, "status": "완료"}
        )
        
        return result
    except Exception as e:
        # 에러 처리
        self.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise
```

### 환경 변수 추가

```python
# app/core/config.py
class Settings(BaseSettings):
    new_setting: str = Field("default_value", env="NEW_SETTING")
```
