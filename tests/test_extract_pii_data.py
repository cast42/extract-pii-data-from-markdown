from unittest.mock import Mock

from extract_pii_data_from_markdown.extract_pii_data import extract_pii_data


def test_extract_pii_data():
    # Example sentence with PII data
    sentence = "My name is John Doe and my email is john.doe@example.com."

    # Mock model with a predict_entities method
    mock_model = Mock()
    mock_model.predict_entities.return_value = [
        {"text": "John Doe", "label": "name", "score": 0.98},
        {"text": "john.doe@example.com", "label": "email", "score": 0.95},
    ]

    # Expected output
    expected_output = [
        {"pii_type": "name", "pii_value": "John Doe", "private": True},
        {"pii_type": "email", "pii_value": "john.doe@example.com", "private": True},
    ]

    # Call the function with the mocked model
    result = extract_pii_data(sentence, mock_model)

    # Assertions
    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_extract_pii_data_without_model():
    # Example sentence with PII data
    sentence = "My name is John Doe."

    # Call the function without a model (model is None)
    result = extract_pii_data(sentence, model=None)

    # Expected output should be an empty list since model is None
    assert result == [], f"Expected an empty list, but got {result}"
