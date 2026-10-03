import socket

import pytest


@pytest.fixture(autouse=True)
def no_provider_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("The release test suite must remain offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
