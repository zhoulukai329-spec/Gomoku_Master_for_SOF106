"""Lightweight tactical move selection for deterministic Gomoku play."""

import math

import numpy as np

from gomoku_game import GomokuGame


DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
WIN_SCORE = 1_000_000.0

# Patterns are ordered from most urgent to least urgent. A match only counts
# when the newly placed stone is inside the matching segment.
PATTERN_SCORES = (
    ("XXXXX", WIN_SCORE),
    (".XXXX.", 260_000.0),
    ("XXXX.", 90_000.0),
    (".XXXX", 90_000.0),
    ("XXX.X", 80_000.0),
    ("XX.XX", 80_000.0),
    ("X.XXX", 80_000.0),
    (".XXX.", 8_000.0),
    (".XX.X.", 6_500.0),
    (".X.XX.", 6_500.0),
    ("XXX..", 1_800.0),
    ("..XXX", 1_800.0),
    ("XX.X.", 1_500.0),
    (".X.XX", 1_500.0),
    (".XX.", 500.0),
)


def choose_strategic_move(
    board,
    current_player,
    legal_moves,
    policy_logits=None,
    policy_weight=1.0,
    heuristic_weight=1.35,
):
    """Return a tactically safer move for deterministic/evaluation play."""
    if not legal_moves:
        return None

    board = np.asarray(board)
    legal_moves = list(legal_moves)

    # Empty-board play should start from the center instead of trusting random
    # initial logits.
    if not np.any(board):
        center = board.shape[0] // 2
        if (center, center) in legal_moves:
            return center, center

    own_wins = find_immediate_wins(board, current_player, legal_moves)
    if own_wins:
        return _best_by_heuristic(board, current_player, own_wins)

    opponent = 3 - current_player
    opponent_wins = set(find_immediate_wins(
        board,
        opponent,
        _legal_moves_for_player(board, opponent),
    ))
    blocking_moves = [move for move in legal_moves if move in opponent_wins]
    if blocking_moves:
        return _best_by_heuristic(board, current_player, blocking_moves)

    heuristic_scores = np.array(
        [score_move(board, move, current_player) for move in legal_moves],
        dtype=np.float64,
    )

    if policy_logits is None:
        return legal_moves[int(np.argmax(heuristic_scores))]

    policy_scores = np.array(
        [float(policy_logits[row * board.shape[0] + col]) for row, col in legal_moves],
        dtype=np.float64,
    )
    combined = (
        policy_weight * _normalize(policy_scores)
        + heuristic_weight * _normalize(heuristic_scores)
    )

    # Keep the heuristic score as a deterministic tie-breaker.
    best_index = max(
        range(len(legal_moves)),
        key=lambda index: (combined[index], heuristic_scores[index]),
    )
    return legal_moves[best_index]


def find_immediate_wins(board, player, candidate_moves):
    """List candidate moves that win immediately for the given player."""
    wins = []
    for row, col in candidate_moves:
        if _move_wins(board, row, col, player):
            wins.append((row, col))
    return wins


def score_move(board, move, player):
    """Score one legal move using attack, defense, center, and locality."""
    row, col = move
    opponent = 3 - player

    attack = _pattern_score(board, row, col, player)
    defense = _pattern_score(board, row, col, opponent)
    center = _center_score(board.shape[0], row, col)
    locality = _locality_score(board, row, col)

    return attack + defense * 0.86 + center + locality


def _best_by_heuristic(board, player, moves):
    scores = [score_move(board, move, player) for move in moves]
    return moves[int(np.argmax(scores))]


def _legal_moves_for_player(board, player):
    game = GomokuGame(board.shape[0])
    game.board = np.array(board, dtype=int, copy=True)
    game.current_player = player
    game.last_move = None
    game.history = []
    game.winner = 0
    return game.get_legal_moves()


def _move_wins(board, row, col, player):
    if board[row, col] != 0:
        return False

    for dr, dc in DIRECTIONS:
        count = 1
        count += _count_direction(board, row, col, player, dr, dc)
        count += _count_direction(board, row, col, player, -dr, -dc)
        if player == 1 and count == 5:
            return True
        if player == 2 and count >= 5:
            return True
    return False


def _count_direction(board, row, col, player, dr, dc):
    total = 0
    size = board.shape[0]
    r, c = row + dr, col + dc
    while 0 <= r < size and 0 <= c < size and board[r, c] == player:
        total += 1
        r += dr
        c += dc
    return total


def _pattern_score(board, row, col, player):
    if board[row, col] != 0:
        return -math.inf

    temp = np.array(board, copy=True)
    temp[row, col] = player

    score = 0.0
    for dr, dc in DIRECTIONS:
        score += _line_score(temp, row, col, player, dr, dc)
    return score


def _line_score(board, row, col, player, dr, dc):
    center_index = 4
    cells = []
    opponent = 3 - player
    size = board.shape[0]

    for offset in range(-4, 5):
        r = row + offset * dr
        c = col + offset * dc
        if not (0 <= r < size and 0 <= c < size):
            cells.append("O")
        elif board[r, c] == player:
            cells.append("X")
        elif board[r, c] == opponent:
            cells.append("O")
        else:
            cells.append(".")

    line = "".join(cells)
    best = 0.0
    for pattern, value in PATTERN_SCORES:
        pattern_len = len(pattern)
        for start in range(0, len(line) - pattern_len + 1):
            end = start + pattern_len
            if start <= center_index < end and line[start:end] == pattern:
                best = max(best, value)
    return best


def _center_score(size, row, col):
    center = (size - 1) / 2.0
    distance = abs(row - center) + abs(col - center)
    return max(0.0, size - distance) * 3.0


def _locality_score(board, row, col):
    size = board.shape[0]
    score = 0.0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            r = row + dr
            c = col + dc
            if 0 <= r < size and 0 <= c < size and board[r, c] != 0:
                score += 8.0 / (abs(dr) + abs(dc))
    return score


def _normalize(values):
    spread = values.std()
    if spread < 1e-9:
        return np.zeros_like(values)
    return (values - values.mean()) / spread
