"""
JobKorea AI Challenge - FastAPI 메인 애플리케이션
"""
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
import structlog
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List
from datetime import datetime

from app.core.config import settings
from app.api.resume import router as resume_router
from app.db.base import async_engine, Base, get_db
from app.services.gemini_service import gemini_service
from app.workers.celery_app import celery_app
from celery.result import AsyncResult


# 로깅 설정
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    logger.info("JobKorea AI Challenge 애플리케이션 시작", version=settings.app_name)
    
    # 데이터베이스 테이블 생성
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("데이터베이스 테이블 생성 완료")
    except Exception as e:
        logger.error("데이터베이스 테이블 생성 실패", error=str(e))
    
    # Gemini 서비스 초기화 확인
    try:
        await gemini_service.health_check()
        logger.info("Gemini AI 서비스 연결 확인 완료")
    except Exception as e:
        logger.warning("Gemini AI 서비스 연결 확인 실패", error=str(e))
    
    # Celery 워커 상태 확인
    try:
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        if active_workers:
            logger.info("Celery 워커 활성화됨", workers=len(active_workers))
        else:
            logger.warning("활성 Celery 워커가 없습니다")
    except Exception as e:
        logger.warning("Celery 워커 상태 확인 실패", error=str(e))
    
    yield
    
    # 종료 시
    logger.info("JobKorea AI Challenge 애플리케이션 종료")
    
    # 리소스 정리
    try:
        await async_engine.dispose()
        logger.info("데이터베이스 연결 정리 완료")
    except Exception as e:
        logger.error("데이터베이스 연결 정리 실패", error=str(e))


# FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.app_name,
    description="JobKorea AI Challenge - 이력서 분석 및 학습 경로 추천 시스템",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts.split(",")
)

# Gzip 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """요청 처리 시간 측정 및 헤더 추가"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청 로깅"""
    start_time = time.time()
    
    # 요청 정보 로깅
    logger.info(
        "요청 시작",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 응답 정보 로깅
        logger.info(
            "요청 완료",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "요청 처리 실패",
            method=request.method,
            url=str(request.url),
            error=str(e),
            process_time=process_time
        )
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리"""
    logger.error(
        "전역 예외 발생",
        method=request.method,
        url=str(request.url),
        error=str(exc),
        error_type=type(exc).__name__
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "내부 서버 오류가 발생했습니다.",
            "detail": str(exc) if settings.debug else "서버 오류가 발생했습니다."
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 처리"""
    logger.warning(
        "HTTP 예외 발생",
        method=request.method,
        url=str(request.url),
        status_code=exc.status_code,
        detail=exc.detail
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


# API 라우터 등록
app.include_router(resume_router, prefix="/api/v1/resume", tags=["resume"])


# 헬스 체크 엔드포인트
@app.get("/health", tags=["default"])
async def health_check() -> Dict[str, Any]:
    """애플리케이션 상태 확인"""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": "1.0.0",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
