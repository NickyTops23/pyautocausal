from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI application
from app.main import app

# Re-use the TestClient fixture style already used in other test modules
client = TestClient(app)


def _load_expected_from_yaml() -> dict:
    """Utility: load the same YAML file that the application uses so that tests
    remain in sync with the deployed configuration."""
    yaml_path = Path(__file__).parent.parent / "app" / "deployed_pipelines.yaml"
    if not yaml_path.exists():
        pytest.skip("deployed_pipelines.yaml not found – skipping pipelines endpoint tests")

    import yaml  # Local import to avoid hard dependency for non-API tests

    with yaml_path.open("r") as f:
        raw_yaml = yaml.safe_load(f)

    pipelines_dict: dict = {}
    if isinstance(raw_yaml, list):
        for entry in raw_yaml:
            pipelines_dict.update(entry)
    elif isinstance(raw_yaml, dict):
        pipelines_dict = raw_yaml
    return pipelines_dict


# -------------------
# Tests
# -------------------

def test_pipelines_endpoint_status_code():
    """Basic check that the /pipelines endpoint responds with HTTP 200."""
    response = client.get("/pipelines")
    assert response.status_code == 200, response.text


def test_pipelines_endpoint_structure():
    """Ensure the endpoint returns a mapping with required and optional columns."""
    response = client.get("/pipelines")
    payload = response.json()

    # Response must be a dict
    assert isinstance(payload, dict)

    expected_from_yaml = _load_expected_from_yaml()

    # The keys of the payload should match the pipeline names in YAML
    assert set(payload.keys()) == set(expected_from_yaml.keys())

    # Each pipeline entry should have required_columns and optional_columns lists
    for pipeline_name, pipeline_config in expected_from_yaml.items():
        assert pipeline_name in payload
        data = payload[pipeline_name]

        assert "required_columns" in data, f"{pipeline_name} missing required_columns"
        assert "optional_columns" in data, f"{pipeline_name} missing optional_columns"

        # required_columns from endpoint should match those from YAML (order not significant)
        assert set(data["required_columns"]) == set(pipeline_config.get("required_columns", []))

        # optional_columns from endpoint should match
        assert set(data["optional_columns"]) == set(pipeline_config.get("optional_columns", [])) 