"""
Gemini 서비스 테스트
"""
import pytest
from unittest.mock import Mock, patch
from app.services.gemini_service import GeminiService


@pytest.fixture
def gemini_service():
    """Gemini 서비스 픽스처"""
    return GeminiService()


def test_create_interview_questions_prompt(gemini_service):
    """면접 질문 프롬프트 생성 테스트"""
    prompt = gemini_service._create_interview_questions_prompt(
        "3년차 백엔드 개발자",
        "Spring Boot 서비스 개발", 
        "Java, Python, MySQL"
    )
    
    assert "3년차 백엔드 개발자" in prompt
    assert "Spring Boot 서비스 개발" in prompt
    assert "Java, Python, MySQL" in prompt
    assert "JSON" in prompt


def test_create_learning_path_prompt(gemini_service):
    """학습 경로 프롬프트 생성 테스트"""
    prompt = gemini_service._create_learning_path_prompt(
        "3년차 백엔드 개발자",
        "Spring Boot 서비스 개발",
        "Java, Python, MySQL"
    )
    
    assert "3년차 백엔드 개발자" in prompt
    assert "학습 경로" in prompt
    assert "JSON" in prompt


def test_parse_text_response_to_questions(gemini_service):
    """텍스트 응답을 질문으로 파싱 테스트"""
    text_response = """
    질문 1: Spring Boot에서 어떤 패턴을 사용하셨나요?
    질문 2: 데이터베이스 최적화는 어떻게 하셨나요?
    """
    
    questions = gemini_service._parse_text_response_to_questions(text_response)
    
    assert len(questions) >= 2
    assert any("Spring Boot" in q["question_text"] for q in questions)
    assert any("데이터베이스" in q["question_text"] for q in questions)


def test_parse_text_response_to_learning_paths(gemini_service):
    """텍스트 응답을 학습 경로로 파싱 테스트"""
    text_response = """
    학습 경로 1: 마이크로서비스 아키텍처 심화
    학습 경로 2: 클라우드 네이티브 기술 습득
    """
    
    paths = gemini_service._parse_text_response_to_learning_paths(text_response)
    
    assert len(paths) >= 2
    assert any("마이크로서비스" in p["title"] for p in paths)
    assert any("클라우드" in p["title"] for p in paths)


@pytest.mark.asyncio
@patch('app.services.gemini_service.genai')
async def test_generate_interview_questions_success(mock_genai, gemini_service):
    """면접 질문 생성 성공 테스트"""
    # Mock 응답 설정
    mock_response = Mock()
    mock_response.text = '''
    {
        "questions": [
            {
                "question_text": "Spring Boot에서 어떤 패턴을 사용하셨나요?",
                "category": "기술역량",
                "difficulty": "중급",
                "reasoning": "실무 경험 확인"
            }
        ]
    }
    '''
    
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    
    questions = await gemini_service.generate_interview_questions(
        "3년차 개발자", "백엔드 개발", "Spring Boot"
    )
    
    assert len(questions) == 1
    assert questions[0]["question_text"] == "Spring Boot에서 어떤 패턴을 사용하셨나요?"
    assert questions[0]["category"] == "기술역량"


@pytest.mark.asyncio
@patch('app.services.gemini_service.genai')
async def test_generate_learning_path_success(mock_genai, gemini_service):
    """학습 경로 생성 성공 테스트"""
    # Mock 응답 설정
    mock_response = Mock()
    mock_response.text = '''
    {
        "learning_paths": [
            {
                "title": "MSA 아키텍처 심화",
                "description": "마이크로서비스 전문성 강화",
                "priority": "높음",
                "estimated_duration": "2-3개월",
                "difficulty": "고급",
                "action_items": [],
                "recommended_resources": [],
                "expected_outcomes": []
            }
        ]
    }
    '''
    
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
    
    paths = await gemini_service.generate_learning_path(
        "3년차 개발자", "백엔드 개발", "Spring Boot"
    )
    
    assert len(paths) == 1
    assert paths[0]["title"] == "MSA 아키텍처 심화"
    assert paths[0]["priority"] == "높음"
