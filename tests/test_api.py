from fastapi.testclient import TestClient
import pytest

from backend.app.main import app
from backend.app.sessions import _GAMES

client = TestClient(app)


def setup_function() -> None:
    _GAMES.clear()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_list_agents_random_available():
    agents = client.get("/agents").json()
    random_agent = next(a for a in agents if a["id"] == "random")
    assert random_agent["available"] is True
    assert next(a for a in agents if a["id"] == "heuristic")["available"] is True
    assert next(a for a in agents if a["id"] == "minimax")["available"] is True
    assert next(a for a in agents if a["id"] == "mcts")["available"] is True
    assert next(a for a in agents if a["id"] == "dqn")["available"] is True
    assert not next(a for a in agents if a["id"] == "alphazero")["available"]


def test_new_game_and_human_move():
    game = client.post("/game/new", json={"agent_id": "random", "first_player": "human"}).json()
    assert game["human_player"] == 1
    assert game["current_player"] == 1
    moved = client.post(f"/game/{game['id']}/move", json={"column": 3}).json()
    assert moved["board"][5][3] == 1
    assert moved["current_player"] == -1


def test_ai_plays_legal_column():
    game = client.post("/game/new", json={"agent_id": "random", "first_player": "human"}).json()
    client.post(f"/game/{game['id']}/move", json={"column": 0})
    after = client.post(f"/game/{game['id']}/ai-move").json()
    assert after["last_ai"] is not None
    assert after["last_action"] in range(7)
    assert after["board"][5][after["last_action"]] == -1 or after["last_action"] == 0


def test_unknown_agent():
    res = client.post("/game/new", json={"agent_id": "alphazero", "first_player": "human"})
    assert res.status_code == 400


def test_dqn_ai_plays_legal_column():
    pytest.importorskip("torch")
    game = client.post("/game/new", json={"agent_id": "dqn", "first_player": "human"}).json()
    assert game["agent_id"] == "dqn"
    client.post(f"/game/{game['id']}/move", json={"column": 3})
    after = client.post(f"/game/{game['id']}/ai-move").json()
    assert after["last_ai"] is not None
    assert after["last_action"] in range(7)
    assert after["last_ai"]["metadata"]["policy"] == "dqn_greedy"


def test_minimax_depth_is_honored():
    game = client.post(
        "/game/new",
        json={"agent_id": "minimax", "first_player": "human", "depth": 3},
    ).json()
    assert game["agent_id"] == "minimax"
    assert game["agent_depth"] == 3
    client.post(f"/game/{game['id']}/move", json={"column": 3})
    after = client.post(f"/game/{game['id']}/ai-move").json()
    assert after["last_ai"]["metadata"]["depth"] == 3


def test_minimax_default_depth():
    game = client.post(
        "/game/new",
        json={"agent_id": "minimax", "first_player": "human"},
    ).json()
    assert game["agent_depth"] == 5


def test_depth_ignored_for_random():
    game = client.post(
        "/game/new",
        json={"agent_id": "random", "first_player": "human", "depth": 7},
    ).json()
    assert game["agent_depth"] is None

