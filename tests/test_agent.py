"""Unit tests for the agentic query router."""
from app.agent import _route, _score_confidence, check_available_slots


class TestRouting:
    """Verify keyword-based route detection."""

    def test_appointment_keywords(self):
        assert _route("Can I book an appointment?") == "appointment"
        assert _route("What slots are available for next week?") == "appointment"
        assert _route("I want to schedule a visit") == "appointment"
        assert _route("How do I reschedule my appointment?") == "appointment"
        assert _route("Cancel my visit") == "appointment"

    def test_drug_recall_keywords(self):
        assert _route("Is Metformin subject to an FDA recall?") == "drug_recall"
        assert _route("Tell me about recalled medications") == "drug_recall"
        assert _route("Which manufacturer had a drug recall?") == "drug_recall"

    def test_healthcare_rag_default(self):
        assert _route("What are the signs of wound infection?") == "healthcare_rag"
        assert _route("Tell me about HIPAA guidelines") == "healthcare_rag"
        assert _route("What is the medication refill policy?") == "healthcare_rag"
        assert _route("How do I request medical records?") == "healthcare_rag"


class TestConfidence:
    """Verify confidence scoring based on document count."""

    def test_high_confidence(self):
        assert _score_confidence([{}, {}, {}]) == "high"
        assert _score_confidence([{}, {}, {}, {}]) == "high"

    def test_medium_confidence(self):
        assert _score_confidence([{}]) == "medium"
        assert _score_confidence([{}, {}]) == "medium"

    def test_low_confidence(self):
        assert _score_confidence([]) == "low"


class TestAppointmentTool:
    """Verify mock appointment slot data."""

    def test_returns_slot_data(self):
        result = check_available_slots()
        assert "Available appointment slots" in result

    def test_contains_doctors(self):
        result = check_available_slots()
        assert "Dr. Sarah Mitchell" in result
        assert "Dr. James Patel" in result
        assert "Dr. Anna Nguyen" in result

    def test_contains_contact_info(self):
        result = check_available_slots()
        assert "(555) 800-1234" in result
