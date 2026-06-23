"""Pygame GUI for human vs neural-network Gomoku."""

import argparse
import sys
import time

import pygame

from agent import GomokuAgent
from bootstrap_training import DEFAULT_BOOTSTRAP_EPISODES, ensure_weights_for_play
from gomoku_game import GomokuGame
from storage import DEFAULT_RUN_NAME, resolve_weights_path

# Fixed layout values keep drawing code easy to read and adjust.
BOARD_SIZE   = 15
GRID_WIDTH   = 40
BOARD_MARGIN = 40
SIDEBAR_W    = 220
WIN_W = BOARD_SIZE * GRID_WIDTH + BOARD_MARGIN * 2 + SIDEBAR_W
WIN_H = BOARD_SIZE * GRID_WIDTH + BOARD_MARGIN * 2

# Centralize all UI colors in one place.
C_BG       = (235, 235, 220)
C_BOARD    = (220, 179, 92)
C_GRID     = (0,   0,   0)
C_BLACK    = (20,  20,  20)
C_WHITE    = (255, 255, 255)
C_RED      = (220,  30,  30)
C_BTN      = (80,  80,  80)
C_BTN_HV   = (120, 120, 120)
C_BTN_TXT  = (255, 255, 255)
C_STATUS   = (40,  40,  40)
C_LABEL    = (60,  60,  60)
C_STAR     = (0,   0,   0)
C_LAST     = (220,  60,  60)
C_THINKING = (200, 100,  30)


class Button:
    """Simple clickable sidebar button."""

    def __init__(self, x, y, w, h, text, cb, active=True):
        self.rect   = pygame.Rect(x, y, w, h)
        self.text   = text
        self.cb     = cb
        self.hov    = False
        self.active = active

    def draw(self, surf, font):
        """Draw the button with hover and disabled states."""
        if not self.active:
            pygame.draw.rect(surf, (160, 160, 160), self.rect, border_radius=6)
        else:
            pygame.draw.rect(surf, C_BTN_HV if self.hov else C_BTN,
                             self.rect, border_radius=6)
        pygame.draw.rect(surf, C_GRID, self.rect, 1, border_radius=6)
        ts = font.render(self.text, True, C_BTN_TXT)
        surf.blit(ts, ts.get_rect(center=self.rect.center))

    def handle(self, ev):
        """Update hover state and run the callback when clicked."""
        if ev.type == pygame.MOUSEMOTION:
            self.hov = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and self.hov and self.active:
            self.cb()


class GomokuGUI:
    """Drive the full human-versus-AI game window."""

    def __init__(self, weights_path=""):
        # Initialize Pygame and create the main window.
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Gomoku AI - Neural PPO")
        self.clock  = pygame.time.Clock()

        # Prefer a CJK-capable font, then fall back to the default font.
        try:
            self.font     = pygame.font.SysFont("SimHei",  22)
            self.big_font = pygame.font.SysFont("SimHei",  38)
            self.sm_font  = pygame.font.SysFont("SimHei",  17)
        except Exception:
            self.font     = pygame.font.SysFont(None, 24)
            self.big_font = pygame.font.SysFont(None, 42)
            self.sm_font  = pygame.font.SysFont(None, 18)

        # Keep one live game object for the current match.
        self.game = GomokuGame(BOARD_SIZE)

        # Resolve the model path before the UI starts interacting with it.
        self.weights_path = resolve_weights_path(
            weights_path=weights_path,
            run_name=DEFAULT_RUN_NAME,
        )
        self.player_col = 1
        self.ai_col = 2
        self.agent = GomokuAgent(size=BOARD_SIZE)
        self.agent_ready = False
        self.agent_error = ""
        self.reload_weights()

        # Store extra UI state that is not part of the game rules themselves.
        self.game_over   = False
        self.winning_seq = []
        self.thinking    = False

        self._build_ui()

    def reload_weights(self):
        """Load the current model weights and update the UI status text."""
        if not self.weights_path.exists():
            self.agent_ready = False
            self.agent_error = f"Missing weights: {self.weights_path.name}"
            return

        try:
            self.agent.load(self.weights_path)
            self.agent_ready = True
            self.agent_error = f"Loaded: {self.weights_path.name}"
            print(f"[GUI] Loaded weights from {self.weights_path}")
        except Exception as exc:
            # Keep the GUI alive even if the file cannot be loaded.
            self.agent_ready = False
            self.agent_error = f"Load failed: {exc}"
            print(f"[GUI] Could not load model: {exc}")

    def _build_ui(self):
        """Create the sidebar buttons once during startup."""
        self.btns = []
        x = BOARD_SIZE * GRID_WIDTH + BOARD_MARGIN * 2 + 16
        y = 30
        dw, dh, gap = 188, 38, 12

        def add(label, cb):
            # Capture the current layout cursor and append one button.
            self.btns.append(Button(x, y, dw, dh, label, cb))

        add("New Game", self.reset_game);            y += dh + gap
        add("Undo", self.undo_move);                 y += dh + gap
        add("Reload Weights", self.reload_weights);  y += dh + gap
        add(self._first_label(), self.toggle_first); y += dh + gap

        self._btn_first = self.btns[3]

    def _first_label(self):
        """Render the current first-player label."""
        return f"First: {'Player' if self.player_col==1 else 'AI'}"

    def toggle_first(self):
        """Swap first move ownership and restart the match."""
        self.player_col = 3 - self.player_col
        self.ai_col     = 3 - self.player_col
        self._btn_first.text = self._first_label()
        self.reset_game()

    def reset_game(self):
        """Reset the live match and trigger the AI when it starts first."""
        self.game.reset()
        self.game_over   = False
        self.winning_seq = []
        self.agent_error = self.agent_error or ""
        if self.ai_col == 1:
            self._do_ai_move()

    def undo_move(self):
        """Undo one human move and one AI move together."""
        if self.game_over:
            return

        self.game.undo_move()
        self.game.undo_move()
        self.game_over = False
        self.winning_seq = []

    def _do_ai_move(self):
        """Ask the neural agent for one move and apply it to the board."""
        if self.game_over:
            return
        if not self.agent_ready:
            return

        # Draw once before inference so the user sees the thinking state.
        self.thinking = True
        self.draw()
        pygame.display.flip()

        # Abort cleanly if no legal moves remain.
        legal = self.game.get_legal_moves()
        if not legal:
            self.thinking = False
            return

        # Use deterministic low-temperature inference for stronger gameplay.
        move = self.agent.select_action(
            self.game.board,
            self.game.current_player,
            legal,
            temperature=0.05,
            deterministic=True,
            record=False,
        )
        self.thinking = False

        if move is None:
            return

        # Apply the move and collect the win highlight when needed.
        self.game.make_move(*move)
        if self.game.winner != 0:
            self.game_over   = True
            self.winning_seq = self.game.get_winning_sequence()

    def _board_px(self, r, c):
        """Convert board coordinates into screen coordinates."""
        return (BOARD_MARGIN + c * GRID_WIDTH,
                BOARD_MARGIN + r * GRID_WIDTH)

    def draw(self):
        """Render the board, stones, sidebar, and current status."""
        self.screen.fill(C_BG)

        # Draw the wooden board panel behind the grid.
        bw = (BOARD_SIZE - 1) * GRID_WIDTH + 2 * 20
        pygame.draw.rect(self.screen, C_BOARD,
                         (BOARD_MARGIN - 20, BOARD_MARGIN - 20, bw, bw))

        # Draw the 15x15 grid.
        for i in range(BOARD_SIZE):
            pygame.draw.line(self.screen, C_GRID,
                             (BOARD_MARGIN, BOARD_MARGIN + i * GRID_WIDTH),
                             (BOARD_MARGIN + (BOARD_SIZE-1)*GRID_WIDTH,
                              BOARD_MARGIN + i * GRID_WIDTH), 1)
            pygame.draw.line(self.screen, C_GRID,
                             (BOARD_MARGIN + i * GRID_WIDTH, BOARD_MARGIN),
                             (BOARD_MARGIN + i * GRID_WIDTH,
                              BOARD_MARGIN + (BOARD_SIZE-1)*GRID_WIDTH), 1)

        # Draw the traditional star points used on a standard Gomoku board.
        stars = [(3,3),(3,11),(11,3),(11,11),(7,7),
                 (3,7),(7,3),(11,7),(7,11)]
        for r, c in stars:
            pygame.draw.circle(self.screen, C_STAR, self._board_px(r, c), 4)

        # Draw every placed stone and highlight the newest move.
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                v = self.game.board[r, c]
                if v == 0:
                    continue
                px = self._board_px(r, c)
                col = C_BLACK if v == 1 else C_WHITE
                radius = GRID_WIDTH // 2 - 2
                pygame.draw.circle(self.screen, col, px, radius)

                # Add a thin border so black and white stones stay readable.
                border = C_WHITE if v == 1 else C_BLACK
                pygame.draw.circle(self.screen, border, px, radius, 1)

                # Mark the newest move so the current line is easy to follow.
                if self.game.last_move == (r, c):
                    mc = C_WHITE if v == 1 else C_BLACK
                    pygame.draw.circle(self.screen, mc, px, 5)

        # Highlight the winning line after the game ends.
        for r, c in self.winning_seq:
            pygame.draw.circle(self.screen, C_RED,
                               self._board_px(r, c),
                               GRID_WIDTH // 2 - 2, 3)

        # Draw the sidebar area that holds controls and status text.
        sx = BOARD_SIZE * GRID_WIDTH + BOARD_MARGIN * 2 + 10

        # Separate the board from the sidebar visually.
        pygame.draw.line(self.screen, C_GRID,
                         (sx - 6, 0), (sx - 6, WIN_H), 1)

        for btn in self.btns:
            btn.draw(self.screen, self.font)

        # Show the highest-priority status for the current frame.
        if self.thinking:
            msg = "Neural agent thinking..."
            ts = self.font.render(msg, True, C_THINKING)
            self.screen.blit(ts, (sx + 4, WIN_H - 120))
        elif self.game_over:
            if self.game.winner == 3:
                msg = "Draw!"
            elif self.game.winner == 1:
                msg = "Black wins!"
            else:
                msg = "White wins!"
            ts = self.big_font.render(msg, True, C_STATUS)
            self.screen.blit(ts, ts.get_rect(
                centerx=sx + 100, y=WIN_H - 130))
        elif not self.agent_ready:
            ts = self.font.render("Weights unavailable", True, C_RED)
            self.screen.blit(ts, (sx + 4, WIN_H - 120))
        else:
            turn = "Black" if self.game.current_player == 1 else "White"
            ts = self.font.render(f"{turn}'s turn", True, C_STATUS)
            self.screen.blit(ts, (sx + 4, WIN_H - 120))

        # Show the move count and active weights file for debugging and clarity.
        mc = self.sm_font.render(f"Moves: {len(self.game.history)}", True, C_LABEL)
        self.screen.blit(mc, (sx + 4, WIN_H - 60))

        weights_line = self.sm_font.render(self.weights_path.name, True, C_LABEL)
        self.screen.blit(weights_line, (sx + 4, WIN_H - 92))
        status_line = self.sm_font.render(self.agent_error[:24], True, C_LABEL)
        self.screen.blit(status_line, (sx + 4, WIN_H - 76))

        pygame.display.flip()

    def run(self):
        """Process Pygame events and keep the window responsive."""
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Live mode lets the human place a stone with a left click.
                if (not self.game_over
                        and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1):
                    mx, my = ev.pos
                    half = GRID_WIDTH // 2

                    # Accept clicks only inside the board area.
                    in_board = (BOARD_MARGIN - half <= mx <= BOARD_MARGIN + (BOARD_SIZE-1)*GRID_WIDTH + half and
                                BOARD_MARGIN - half <= my <= BOARD_MARGIN + (BOARD_SIZE-1)*GRID_WIDTH + half)
                    if in_board and self.game.current_player == self.player_col:
                        c = round((mx - BOARD_MARGIN) / GRID_WIDTH)
                        r = round((my - BOARD_MARGIN) / GRID_WIDTH)
                        ok, _ = self.game.make_move(r, c)
                        if ok:
                            # Either finish the game now or hand the turn to the AI.
                            if self.game.winner:
                                self.game_over   = True
                                self.winning_seq = self.game.get_winning_sequence()
                            else:
                                self.draw()
                                pygame.display.flip()
                                time.sleep(0.05)
                                self._do_ai_move()

                # Sidebar buttons remain active throughout the live match.
                for btn in self.btns:
                    btn.handle(ev)

            self.draw()
            self.clock.tick(30)


def parse_args():
    """Parse the optional weights path passed from the command line."""
    parser = argparse.ArgumentParser(description="Launch the Gomoku GUI")
    parser.add_argument("--weights-path", type=str, default="")
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--bootstrap-episodes",
        type=int,
        default=DEFAULT_BOOTSTRAP_EPISODES,
        help="Self-play episodes to run before the GUI when weights are missing",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Run self-play training before opening the GUI even if weights exist",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Open the GUI without automatic self-play training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    weights_path = args.weights_path
    if not args.skip_training:
        weights_path, trained = ensure_weights_for_play(
            weights_path=args.weights_path,
            run_name=args.run_name,
            size=BOARD_SIZE,
            episodes=args.bootstrap_episodes,
            force_train=args.force_train,
        )
        if trained:
            print(f"[Bootstrap] Ready for play with weights: {weights_path}")

    gui = GomokuGUI(weights_path=weights_path)
    gui.run()
