import pytest, httpx
from unittest.mock import AsyncMock, MagicMock, patch

async def test_check_http_service_up():
    from admin.health import check_http_service
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("admin.health.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value.get = AsyncMock(return_value=mock_resp)
        result = await check_http_service("chatbot", "http://chatbot:8000/health")
    assert result["name"] == "chatbot" and result["healthy"] is True and "latency_ms" in result

async def test_check_http_service_down():
    from admin.health import check_http_service
    with patch("admin.health.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__ = AsyncMock(return_value=MockClient.return_value)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        result = await check_http_service("chatbot", "http://chatbot:8000/health")
    assert result["healthy"] is False and "error" in result
