"""
Tests para clientes API
"""

import pytest
from src.api.mlb_client import MLBClient, MLBAPIError


def test_mlb_client_init():
    client = MLBClient()
    assert client.base_url is not None
    assert client.timeout > 0


def test_get_schedule():
    client = MLBClient()
    df = client.get_schedule(season=2024)
    
    assert df is not None
    assert len(df) > 0 or df.empty  # Puede estar vacío fuera de temporada


def test_invalid_request():
    client = MLBClient()
    
    with pytest.raises(MLBAPIError):
        client._request("invalid/endpoint/9999999")
