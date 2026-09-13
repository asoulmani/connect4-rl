"""Stateless game API tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend" / "app"


def _context(game: dict, **extra):
    body = {
        "moves": game["moves"],
        "human_player": game["human_player"],
        "agent_id": game["agent_id"],
        "agent_depth": game["agent_depth"],
    }
    body.update(extra)
    return body


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


def test_new_game_human_starts_empty():
    game = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    assert "id" not in game
    assert game["moves"] == []
    assert game["human_player"] == 1
    assert game["current_player"] == 1
    assert game["done"] is False
    assert game["board"] == [[0] * 7 for _ in range(6)]
    assert game["last_action"] is None
    assert game["last_ai"] is None


def test_new_game_ai_first_includes_opening_move():
    game = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "ai"}
    ).json()
    assert game["human_player"] == -1
    assert len(game["moves"]) == 1
    col = game["moves"][0]
    assert col in range(7)
    assert game["last_action"] == col
    assert game["last_ai"] is not None
    assert game["board"][5][col] == 1  # PLAYER_ONE opened
    assert game["current_player"] == -1


def test_human_move_applies_column():
    game = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    moved = client.post("/game/move", json=_context(game, column=3)).json()
    assert moved["moves"] == [3]
    assert moved["board"][5][3] == 1
    assert moved["current_player"] == -1
    assert moved["last_action"] == 3


def test_ai_move_appends_one_move():
    game = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    after_human = client.post("/game/move", json=_context(game, column=0)).json()
    after = client.post("/game/ai-move", json=_context(after_human)).json()
    assert len(after["moves"]) == 2
    assert after["moves"][0] == 0
    assert after["last_ai"] is not None
    assert after["last_action"] in range(7)
    assert after["last_action"] == after["moves"][1]


def test_independent_games_do_not_interfere():
    a = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    b = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    a2 = client.post("/game/move", json=_context(a, column=0)).json()
    b2 = client.post("/game/move", json=_context(b, column=6)).json()
    assert a2["moves"] == [0]
    assert b2["moves"] == [6]
    assert a2["board"][5][0] == 1
    assert a2["board"][5][6] == 0
    assert b2["board"][5][6] == 1
    assert b2["board"][5][0] == 0


def test_invalid_history_column_out_of_range():
    game = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    res = client.post(
        "/game/move",
        json={**_context(game), "moves": [99], "column": 0},
    )
    assert res.status_code == 400


def test_history_after_terminal_rejected():
    # Vertical win for player 1 in column 0; extra move after terminal.
    # Moves: P1 col0, P2 col1, P1 col0, P2 col1, P1 col0, P2 col1, P1 col0 (win)
    winning = [0, 1, 0, 1, 0, 1, 0]
    game = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    res = client.post(
        "/game/ai-move",
        json={
            "moves": winning + [2],
            "human_player": game["human_player"],
            "agent_id": "random",
            "agent_depth": None,
        },
    )
    assert res.status_code == 400
    assert "finished" in res.json()["detail"].lower() or "after" in res.json()["detail"].lower()


def test_full_column_in_history_rejected():
    # Six stones in column 0 fills it; seventh in history is illegal.
    fill = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # After 12 moves col0 has 6 and col1 has 6 — both full. Use only col0 overfill:
    # Alternate with side columns so col0 gets 7 attempts... simpler: 7 drops in same
    # column can't happen with alternating players unless we force illegal mid-history.
    # Build: fill column 0 completely (6 plies) by pairing with column 1, then add 0 again.
    # After [0,1,0,1,0,1,0,1,0,1,0,1] both cols full. History + [0] is illegal.
    game = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    res = client.post(
        "/game/move",
        json={
            "moves": fill + [0],
            "human_player": 1,
            "agent_id": "random",
            "agent_depth": None,
            "column": 2,
        },
    )
    assert res.status_code == 400


def test_no_global_game_registry_in_backend():
    forbidden = {"_GAMES", "GameSession", "get_game"}
    found: list[str] = []
    for path in BACKEND_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden:
                found.append(f"{path.name}:{node.id}")
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                found.append(f"{path.name}:{node.attr}")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in forbidden:
                        found.append(f"{path.name}:{target.id}")
    assert found == []


def test_unknown_agent():
    res = client.post("/game/new", json={"agent_id": "alphazero", "first_player": "human"})
    assert res.status_code == 400


def test_dqn_ai_plays_legal_column():
    pytest.importorskip("torch")
    game = client.post(
        "/game/new", json={"agent_id": "dqn", "first_player": "human"}
    ).json()
    assert game["agent_id"] == "dqn"
    after_human = client.post("/game/move", json=_context(game, column=3)).json()
    after = client.post("/game/ai-move", json=_context(after_human)).json()
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
    after_human = client.post("/game/move", json=_context(game, column=3)).json()
    after = client.post("/game/ai-move", json=_context(after_human)).json()
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


def test_wrong_turn_rejected():
    game = client.post(
        "/game/new", json={"agent_id": "random", "first_player": "human"}
    ).json()
    res = client.post("/game/ai-move", json=_context(game))
    assert res.status_code == 400


def test_get_game_by_id_removed():
    res = client.get("/game/abc123")
    assert res.status_code == 404
