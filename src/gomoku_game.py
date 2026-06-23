import numpy as np


class GomokuGame:
    """Store the board state and enforce all Gomoku and Renju rules."""

    def __init__(self, size=15):
        # remember how big the board is (15x15 is the standard size).
        self.size = size
        # build the very first empty board using the reset logic below,
        # so __init__ and reset() never get out of sync with each other.
        self.reset()

    def reset(self):
        """Clear the board and restore the starting player."""
        # make a fresh size x size grid filled with zeros.
        # 0 = empty cell, 1 = black stone, 2 = white stone.
        self.board = np.zeros((self.size, self.size), dtype=int)
        # black (player 1) always moves first in Gomoku/Renju.
        self.current_player = 1
        # no move has been played yet, so there is no "last move" marker.
        self.last_move = None
        # history stores every (row, col, player) move in order,
        # which is what powers undo_move() and move-count displays in the GUI.
        self.history = []
        # 0 = game still in progress, 1 = black won, 2 = white won, 3 = draw.
        self.winner = 0
        return self.get_state()

    def get_state(self):
        """Return a copy so callers cannot mutate the live board by mistake."""
        # A plain copy() is used here on purpose: if we returned self.board directly,
        # any caller that edits the returned array would silently corrupt the real game.
        return self.board.copy()

    def is_legal(self, row, col):
        """Check board bounds, emptiness, and black forbidden-move rules."""
        # reject any coordinate outside the board.
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        # reject any cell that is already occupied.
        if self.board[row, col] != 0:
            return False

        # only black is restricted by Renju's forbidden-move rules
        # (overline, double-four, double-three). White can play anywhere empty.
        if self.current_player == 1:
            if self.is_forbidden(row, col):
                return False
        return True

    def make_move(self, row, col):
        """Place one stone, update winner state, and switch turns."""
        # refuse the move immediately if it breaks any rule.
        if not self.is_legal(row, col):
            return False, "Illegal move"

        # place the stone and record it as the most recent move.
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.history.append((row, col, self.current_player))

        # a move can end the game two ways: forming a winning line,
        # or being the very last empty cell on a completely filled board (draw).
        if self.check_win(row, col):
            self.winner = self.current_player
        elif len(self.history) == self.size * self.size:
            self.winner = 3

        # hand the turn to the other player (1 -> 2, 2 -> 1) using the
        # "3 minus player" trick, which works because the two player IDs are 1 and 2.
        self.current_player = 3 - self.current_player
        return True, None

    def undo_move(self):
        """Remove the latest move and rebuild the turn state around it."""
        # nothing to undo if no moves have been played.
        if not self.history:
            return False

        # pop the most recent move off the history stack.
        row, col, player = self.history.pop()
        # clear that cell back to empty.
        self.board[row, col] = 0
        # give the turn back to the player whose move we just removed.
        self.current_player = player
        # undoing a move always returns the game to "in progress".
        self.winner = 0
        # recompute last_move from whatever move is now at the top of
        # history, or None if the board is empty again.
        self.last_move = self.history[-1][:2] if self.history else None
        return True

    def check_win(self, row, col):
        """Check whether the last move created a legal winning line."""
        player = self.board[row, col]
        # Only 4 direction vectors are needed: horizontal, vertical, and the
        # two diagonals. Each loop below walks both ways along one direction.
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            # Start the count at 1 to include the stone that was just placed.
            count = 1

            # walk forward along this direction, counting matching stones.
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # walk backward along the same direction (the opposite way).
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            # apply the Renju win condition. Black must form EXACTLY
            # five in a row (six or more is an "overline" and does not win).
            # White has no such restriction and wins with five or more.
            if player == 1:
                if count == 5:
                    return True
            else:
                if count >= 5:
                    return True
        return False

    def is_forbidden(self, row, col):
        """Check whether a black move would violate Renju forbidden rules.
        (Since if the player can make "double three", "double four", or "overline" patterns, 
        the black player will win permanently, 
        so these patterns are forbidden for black to play.)"""
        # temporarily place the stone so the helper checks below can
        # "look into the future" and see what the board would look like.
        self.board[row, col] = 1

        # check the three forbidden patterns one at a time, restoring
        # the board to empty before returning as soon as one pattern matches.

        # Overline: a black line longer than five stones.
        if self._check_overline(row, col):
            self.board[row, col] = 0
            return True

        # An exact five-in-a-row is a legal winning move for black. Check it
        # before double-four and double-three so a real finishing move does
        # not get filtered out of the legal-move list by secondary threats.
        if self._creates_exact_five(row, col):
            self.board[row, col] = 0
            return False

        # Double four: the move creates two separate four-in-a-row threats
        # at once, which is illegal because it is an unstoppable win setup.
        if self._check_double_four(row, col):
            self.board[row, col] = 0
            return True

        # Double three: the move creates two separate open-three threats
        # at once, for the same reason as double four.
        if self._check_double_three(row, col):
            self.board[row, col] = 0
            return True

        # none of the forbidden patterns matched, so undo the
        # temporary placement and tell the caller this move is allowed.
        self.board[row, col] = 0
        return False

    def _check_overline(self, row, col):
        """Return True when the move creates a black line of length six or more."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            # Same forward/backward counting pattern as check_win, but this
            # helper only ever looks at black stones (value == 1).
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
            # Any line longer than five stones is an illegal overline for black.
            if count > 5:
                return True
        return False

    def _creates_exact_five(self, row, col):
        """Return True when black has exactly five stones in one direction."""
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
            if count == 5:
                return True
        return False

    def _check_double_four(self, row, col):
        """Count how many directions become a four-threat after the move."""
        fours = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self._is_four(row, col, dr, dc):
                fours += 1
        # Two or more four-threats from one move means the move is forbidden.
        return fours >= 2

    def _is_four(self, row, col, dr, dc):
        """Check whether the move creates a direct or jump four in one direction."""
        # Start at 1 to count the stone that was just placed.
        count = 1

        # count consecutive black stones going forward, and remember
        # the first empty cell right after them (if any) as a potential
        # extension point for the four-in-a-row.
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
            count += 1
            r += dr
            c += dc
        space1 = (r, c) if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0 else None

        # do the same thing going backward.
        r, c = row - dr, col - dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
            count += 1
            r -= dr
            c -= dc
        space2 = (r, c) if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0 else None

        if count == 4:
            # a straight four (XXXX) is a real threat only if at
            # least one end is still open to place the fifth stone.
            if space1 or space2:
                return True
        elif count == 3:
            # also catch "broken four" shapes such as XXX_X, where
            # the four stones are not all adjacent but still force a win.
            # Check one cell past each open end for another black stone.
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
        # Two or more open-three threats from one move means it is forbidden.
        return threes >= 2

    def _is_open_three(self, row, col, dr, dc):
        """Check a simplified open-three pattern in one direction."""
        # Start at 1 to count the stone that was just placed.
        count = 1
        # count consecutive black stones forward, then require the
        # very next cell to be empty (an "open" end for future extension).
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
            count += 1
            r += dr
            c += dc
        if not (0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0):
            return False

        # same check going backward, requiring an open end there too.
        r, c = row - dr, col - dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 1:
            count += 1
            r -= dr
            c -= dc
        if not (0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0):
            return False

        # only a line of exactly three stones with both ends open
        # counts as an "open three" for this simplified check.
        return count == 3

    def get_legal_moves(self):
        """Return every board cell that is currently legal to play."""
        # Brute-force scan over every cell. For a 15x15 board this is only
        # 225 checks, which is cheap enough to call every turn.
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal(r, c):
                    moves.append((r, c))
        return moves

    def get_winning_sequence(self):
        """Return the line to highlight after the game ends."""
        # there is nothing to highlight if the game is still going
        # or ended in a draw (winner == 3).
        if self.winner == 0 or self.winner == 3:
            return []

        row, col = self.last_move
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        # try each direction and rebuild the exact sequence of
        # coordinates that make up the winning line, for the GUI to draw.
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

            # the first direction that produces 5+ stones is the
            # winning line (overlines for black are already excluded earlier
            # since check_win only sets winner==1 on an exact 5-count).
            if len(seq) >= 5:
                return seq
        return []
