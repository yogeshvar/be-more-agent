from beemboy.guardrails import check_guardrails


def test_blocks_latest_news_query():
    result = check_guardrails("What is the latest news today?")
    assert result.blocked is True
    assert result.response is not None


def test_allows_personal_productivity_query():
    result = check_guardrails("Help me plan tomorrow and check my reminders.")
    assert result.blocked is False
