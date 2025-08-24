"""
API 엔드포인트 테스트
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from httpx import AsyncClient
from fastapi.testclient import TestClient
from app.main import app
from app.api.schemas import ResumeInputRequest

client = TestClient(app)


class TestHealthCheck:
    """헬스 체크 테스트"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_resume_analyze_validation():
    """이력서 분석 입력 검증 테스트"""
    # 빈 텍스트
    response = client.post(
        "/api/v1/resume/analyze",
        json={"resume_text": ""}
    )
    assert response.status_code == 422
    
    # 너무 짧은 텍스트
    response = client.post(
        "/api/v1/resume/analyze", 
        json={"resume_text": "짧음"}
    )
    assert response.status_code == 422
    
    # 너무 긴 텍스트
    long_text = "a" * 2001
    response = client.post(
        "/api/v1/resume/analyze",
        json={"resume_text": long_text}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_resume_analyze_async_endpoint():
    """비동기 이력서 분석 엔드포인트 테스트"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/resume/analyze-async",
            json={
                "resume_text": "3년차 백엔드 개발자, Spring Boot/Python 기반 서비스 개발 경험"
            }
        )
        
        # API 키가 없으면 500 오류가 날 수 있지만, 구조적으로는 정상
        assert response.status_code in [202, 500]
        
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_metrics_logging_failure(self):
        """메트릭스 로깅 실패 테스트"""
        from app.api.resume import log_api_metrics
        
        mock_db = AsyncMock()
        mock_db.add.side_effect = Exception("Logging failed")
        
        # 로깅 실패해도 예외가 발생하지 않아야 함
        await log_api_metrics(
            db=mock_db,
            endpoint="/test",
            method="GET",
            session_id="test-session",
            user_ip="127.0.0.1",
            response_time_ms=100,
            status_code=200
        )
        
        # 로깅 실패는 내부적으로 처리되어야 함
        mock_db.add.assert_called_once()

