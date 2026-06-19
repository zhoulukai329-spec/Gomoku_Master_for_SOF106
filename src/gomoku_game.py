import numpy as np


class GomokuGame:
    """Store the board state and enforce all Gomoku and Renju rules."""

    def __init__(self, size=15):
        # Save the board size once, then build the initial empty game.
        self.size = size
        self.reset()

    def reset(self):
        """Clear the board and restore the starting player."""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.last_move = None
        self.history = []
        self.winner = 0
        return self.get_state()

    def get_state(self):
        """Return a copy so callers cannot mutate the live board by mistake."""
        return self.board.copy()

    def is_legal(self, row, col):
        """Check board bounds, emptiness, and black forbidden-move rules."""
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row, col] != 0:
            return False

        # Only black is restricted by Renju forbidden-move rules.
        if self.current_player == 1:
            if self.is_forbidden(row, col):
                return False
        return True

    def make_move(self, row, col):
        """Place one stone, update winner state, and switch turns."""
        if not self.is_legal(row, col):
            return False, "Illegal move"

        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.history.append((row, col, self.current_player))

        # A move can end the game by win or by filling the final empty cell.
        if self.check_win(row, col):
            self.winner = self.current_player
        elif len(self.history) == self.size * self.size:
            self.winner = 3

        # Toggle between player 1 and player 2 after every accepted move.
        self.current_player = 3 - self.current_player
        return True, None

    def undo_move(self):
        """Remove the latest move and rebuild the turn state around it."""
        if not self.history:
            return False

        row, col, player = self.history.pop()
        self.board[row, col] = 0
        self.current_player = player
        self.winner = 0
        self.last_move = self.history[-1][:2] if self.history else None
        return True

    def check_win(self, row, col):
        """Check whether the last move created a legal winning line."""
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            # Count matching stones forward from the new move.
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # Count matching stones backward from the new move.
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            # Black must make exactly five under Renju. White may exceed five.
            if player == 1:
                if count == 5:
                    return True
            else:
                if count >= 5:
                    return True
        return False

    def is_forbidden(self, row, col):
        """Check whether a black move would violate Renju forbidden rules."""
        # Place the stone temporarily so pattern checks see the future position.
        self.board[row, col] = 1

        # Overline means black made a line longer than five.
        if self._check_overline(row, col):
            self.board[row, col] = 0
            return True

        # Double four means one move creates two separate four threats.
        if self._check_double_four(row, col):
            self.board[row, col] = 0
            return True

        # Double three means one move creates two separate open threes.
        if self._check_double_three(row, col):
            self.board[row, col] = 0
            return True

        # Restore the empty cell before returning to the caller.
        self.board[row, col] = 0
        return False

    def _check_overline(self, row, col):
        """Return True when the move creates a black line of length six or more."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
                count += 1
                r -= dr
                c -= dc
            if count > 5:
                return True
        return False

    def _check_double_four(self, row, col):
        """Count how many directions become a four-threat after the move."""
        fours = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self._is_four(row, col, dr, dc):
                fours += 1
        return fours >= 2

    def _is_four(self, row, col, dr, dc):
        """Check whether the move creates a direct or jump four in one direction."""
        count = 1

        # Count consecutive stones in the forward direction.
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
            count += 1
            r += dr
            c += dc
        space1 = (r, c) if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0 else None

        # Count consecutive stones in the backward direction.
        r, c = row - dr, col - dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
            count += 1
            r -= dr
            c -= dc
        space2 = (r, c) if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0 else None

        if count == 4:
            # A straight four is valid when at least one end stays open.
            if space1 or space2:
                return True
        elif count == 3:
            # Also detect jump-four shapes such as 1101.
            if space1:
                r, c = space1[0] + dr, space1[1] + dc
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
                    return True
            if space2:
                r, c = space2[0] - dr, space2[1] - dc
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
                    return True
        return False

    def _check_double_three(self, row, col):
        """Count how many open-three patterns are created by the move."""
        threes = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self._is_open_three(row, col, dr, dc):
                threes += 1
        return threes >= 2

    def _is_open_three(self, row, col, dr, dc):
        """Check a simplified open-three pattern in one direction."""
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
            count += 1
            r += dr
            c += dc
        if not (0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0):
            return False
        
        r, c = row - dr, col - dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
            count += 1
            r -= dr
            c -= dc
        if not (0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0):
            return False

        return count == 3

    def get_legal_moves(self):
        """Return every board cell that is currently legal to play."""
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal(r, c):
                    moves.append((r, c))
        return moves

    def get_winning_sequence(self):
        """Return the line to highlight after the game ends."""
        if self.winner == 0 or self.winner == 3:
            return []

        row, col = self.last_move
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            seq = [(row, col)]
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                seq.append((r, c))
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                seq.append((r, c))
                r -= dr
                c -= dc

            if len(seq) >= 5:
                return seq
        return []
