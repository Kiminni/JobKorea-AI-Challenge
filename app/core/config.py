"""
애플리케이션 설정 관리
"""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # 기본 설정
    app_name: str = "JobKorea AI Challenge"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = Field("dev-secret-key", env="SECRET_KEY")
    
    # 데이터베이스 설정
    database_url: str = Field("postgresql+asyncpg://jobkorea:jobkorea_pass@postgres:5432/jobkorea_ai", env="DATABASE_URL")
    
    # Redis 설정
    redis_url: str = Field("redis://redis:6379/0", env="REDIS_URL")
    
    # RabbitMQ 설정
    rabbitmq_url: str = Field("amqp://jobkorea:jobkorea_rabbit@rabbitmq:5672/jobkorea_vhost", env="RABBITMQ_URL")
    
    # AI 서비스 설정
    gemini_api_key: str = Field("dummy-key", env="GEMINI_API_KEY")
    gemini_timeout_seconds: int = Field(30, env="GEMINI_TIMEOUT_SECONDS")
    gemini_max_retries: int = Field(3, env="GEMINI_MAX_RETRIES")
    
    # 보안 설정
    allowed_hosts: str = Field("localhost,127.0.0.1", env="ALLOWED_HOSTS")
    cors_origins: str = Field("http://localhost:3000", env="CORS_ORIGINS")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    
    # Celery 설정
    celery_broker_url: str = Field("redis://redis:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field("redis://redis:6379/0", env="CELERY_RESULT_BACKEND")
    celery_task_serializer: str = Field("json", env="CELERY_TASK_SERIALIZER")
    celery_result_serializer: str = Field("json", env="CELERY_RESULT_SERIALIZER")
    celery_accept_content: List[str] = Field(["json"], env="CELERY_ACCEPT_CONTENT")
    celery_timezone: str = Field("Asia/Seoul", env="CELERY_TIMEZONE")
    celery_enable_utc: bool = Field(True, env="CELERY_ENABLE_UTC")
    celery_task_track_started: bool = Field(True, env="CELERY_TASK_TRACK_STARTED")
    celery_task_time_limit: int = Field(300, env="CELERY_TASK_TIME_LIMIT")
    celery_task_soft_time_limit: int = Field(240, env="CELERY_TASK_SOFT_TIME_LIMIT")
    celery_worker_prefetch_multiplier: int = Field(1, env="CELERY_WORKER_PREFETCH_MULTIPLIER")
    celery_task_acks_late: bool = Field(True, env="CELERY_TASK_ACKS_LATE")
    celery_worker_max_tasks_per_child: int = Field(1000, env="CELERY_WORKER_MAX_TASKS_PER_CHILD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 싱글톤 패턴으로 설정 인스턴스 생성
settings = Settings()
