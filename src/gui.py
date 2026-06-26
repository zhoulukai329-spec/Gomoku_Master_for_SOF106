"""Pygame GUI for human vs neural-network Gomoku."""

import argparse
import sys
import time

import pygame

from agent import GomokuAgent
from bootstrap_training import DEFAULT_BOOTSTRAP_EPISODES, ensure_weights_for_play
from gomoku_game import GomokuGame
from runtime_paths import resource_path
from storage import DEFAULT_RUN_NAME, resolve_weights_path

# Fixed layout values keep drawing code easy to read and adjust.
BOARD_SIZE = 15
GRID_WIDTH = 40
BOARD_MARGIN = 40
SIDEBAR_W = 220
WIN_W = BOARD_SIZE * GRID_WIDTH + BOARD_MARGIN * 2 + SIDEBAR_W
WIN_H = BOARD_SIZE * GRID_WIDTH + BOARD_MARGIN * 2

# Earthy / Wood UI Palette
C_BG = (248, 243, 232)
C_BOARD = (220, 175, 105)
C_SHADOW = (220, 210, 195)
C_GRID = (80, 55, 35)
C_BLACK = (35, 35, 35)
C_WHITE = (240, 240, 235)
C_RED = (200, 60, 60)
C_BTN = (130, 95, 65)
C_BTN_HV = (155, 115, 85)
C_BTN_DIS = (205, 190, 175)
C_BTN_BRD = (100, 70, 45)
C_BTN_TXT = (255, 255, 245)
C_STATUS = (0, 0, 0)
C_LABEL = (140, 120, 105)
C_STAR = (80, 55, 35)
C_LAST = (200, 60, 60)
C_THINKING = (210, 110, 30)


class Button:
    """Simple clickable sidebar button."""

    def __init__(self, x, y, w, h, text, cb, active=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.cb = cb
        self.hov = False
        self.active = active

    def draw(self, surf, font):
        """Draw the button with hover and disabled states."""
        if not self.active:
            pygame.draw.rect(surf, C_BTN_DIS, self.rect, border_radius=8)
        else:
            pygame.draw.rect(surf, C_BTN_HV if self.hov else C_BTN,
                             self.rect, border_radius=8)

        if self.active and not self.hov:
            pygame.draw.rect(surf, C_BTN_BRD, self.rect, 2, border_radius=8)

        ts = font.render(self.text, True, C_BTN_TXT)
        surf.blit(ts, ts.get_rect(center=self.rect.center))


    def handle(self, ev):
        """Update hover state and run the callback when clicked."""
        if ev.type == pygame.MOUSEMOTION:
            self.hov = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and self.active and self.rect.collidepoint(ev.pos):
            self.cb()


class GomokuGUI:
    """Drive the full human-versus-AI game window."""

    def __init__(self, weights_path=""):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Gomoku AI")
        self.clock = pygame.time.Clock()

        fonts = "segoeui, helvetica, arial, simhei"
        self.font = pygame.font.SysFont(fonts, 20, bold=True)
        self.big_font = pygame.font.SysFont(fonts, 36, bold=True)
        self.sm_font = pygame.font.SysFont(fonts, 16)

        self.game = GomokuGame(BOARD_SIZE)

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

        self.game_over = False
        self.winning_seq = []
        self.thinking = False

        # --- FITUR BACKGROUND ---

        self.bg_image = None
        try:
            img = pygame.image.load(str(resource_path("bg_china.png"))).convert()
            self.bg_image = pygame.transform.smoothscale(img, (WIN_W, WIN_H))
            self.bg_image.set_alpha(120)
        except Exception as e:
            print(f"[GUI] Info: Gambar background tidak ditemukan ({e}). Menggunakan warna solid.")
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
            self.agent_ready = False
            self.agent_error = f"Load failed: {exc}"
            print(f"[GUI] Could not load model: {exc}")

    def _build_ui(self):
        """Create the sidebar buttons once during startup."""
        self.btns = []
        x = BOARD_SIZE * GRID_WIDTH + BOARD_MARGIN * 2 + 16
        y = 40
        dw, dh, gap = 188, 42, 16

        def add(label, cb):
            self.btns.append(Button(x, y, dw, dh, label, cb))

        add("New Game", self.reset_game);
        y += dh + gap
        add("Undo Move", self.undo_move);
        y += dh + gap
        add("Reload Weights", self.reload_weights);
        y += dh + gap
        add(self._first_label(), self.toggle_first);
        y += dh + gap

        self._btn_first = self.btns[3]

    def _first_label(self):
        return f"First: {'Player' if self.player_col == 1 else 'AI'}"

    def toggle_first(self):
        self.player_col = 3 - self.player_col
        self.ai_col = 3 - self.player_col
        self._btn_first.text = self._first_label()
        self.reset_game()

    def reset_game(self):
        self.game.reset()
        self.game_over = False
        self.winning_seq = []
        self.agent_error = self.agent_error or ""
        if self.ai_col == 1:
            self._do_ai_move()

    def undo_move(self):
        if self.game_over:
            return
        self.game.undo_move()
        self.game.undo_move()
        self.game_over = False
        self.winning_seq = []

    def _do_ai_move(self):
        if self.game_over or not self.agent_ready:
            return

        self.thinking = True
        self.draw()
        pygame.display.flip()

        legal = self.game.get_legal_moves()
        if not legal:
            self.thinking = False
            return

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

        self.game.make_move(*move)
        if self.game.winner != 0:
            self.game_over = True
            self.winning_seq = self.game.get_winning_sequence()

    def _board_px(self, r, c):
        return (BOARD_MARGIN + c * GRID_WIDTH,
                BOARD_MARGIN + r * GRID_WIDTH)

    def draw(self):

        self.screen.fill(C_BG)

        if self.bg_image:
            self.screen.blit(self.bg_image, (0, 0))

        bw = (BOARD_SIZE - 1) * GRID_WIDTH + 2 * 20


        shadow_surf = pygame.Surface((bw, bw), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (*C_SHADOW, 180), shadow_surf.get_rect(), border_radius=12)
        self.screen.blit(shadow_surf, (BOARD_MARGIN - 16, BOARD_MARGIN - 16))


        board_surf = pygame.Surface((bw, bw), pygame.SRCALPHA)
        pygame.draw.rect(board_surf, (*C_BOARD, 235), board_surf.get_rect(), border_radius=12)
        self.screen.blit(board_surf, (BOARD_MARGIN - 20, BOARD_MARGIN - 20))

        # Draw the 15x15 grid
        for i in range(BOARD_SIZE):
            pygame.draw.line(self.screen, C_GRID,
                             (BOARD_MARGIN, BOARD_MARGIN + i * GRID_WIDTH),
                             (BOARD_MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH,
                              BOARD_MARGIN + i * GRID_WIDTH), 1)
            pygame.draw.line(self.screen, C_GRID,
                             (BOARD_MARGIN + i * GRID_WIDTH, BOARD_MARGIN),
                             (BOARD_MARGIN + i * GRID_WIDTH,
                              BOARD_MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH), 1)

        # Draw star points
        stars = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7),
                 (3, 7), (7, 3), (11, 7), (7, 11)]
        for r, c in stars:
            pygame.draw.circle(self.screen, C_STAR, self._board_px(r, c), 4)

        # Draw stones with 3D aesthetic
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                v = self.game.board[r, c]
                if v == 0:
                    continue
                px = self._board_px(r, c)
                radius = GRID_WIDTH // 2 - 3

                col = C_BLACK if v == 1 else C_WHITE
                pygame.draw.circle(self.screen, col, px, radius)

                hl_color = (80, 80, 80) if v == 1 else (255, 255, 255)
                hl_px = (px[0] - 4, px[1] - 4)
                pygame.draw.circle(self.screen, hl_color, hl_px, radius // 2.5)

                if v == 2:
                    pygame.draw.circle(self.screen, (200, 200, 200), px, radius, 1)

                if self.game.last_move == (r, c):
                    mc = C_RED
                    pygame.draw.circle(self.screen, mc, px, 4)

        # Highlight winning line
        for r, c in self.winning_seq:
            pygame.draw.circle(self.screen, C_RED,
                               self._board_px(r, c),
                               GRID_WIDTH // 2 - 2, 3)

        sx = BOARD_SIZE * GRID_WIDTH + BOARD_MARGIN * 2 + 10

        # Draw a soft separator line
        pygame.draw.line(self.screen, (225, 215, 205),
                         (sx - 6, 20), (sx - 6, WIN_H - 20), 2)

        for btn in self.btns:
            btn.draw(self.screen, self.font)

        # Status text
        if self.thinking:
            msg = "AI is thinking..."
            ts = self.font.render(msg, True, C_THINKING)
            self.screen.blit(ts, (sx + 4, WIN_H - 140))
        elif self.game_over:
            if self.game.winner == 3:
                msg = "It's a Draw!"
            elif self.game.winner == 1:
                msg = "Black Wins!"
            else:
                msg = "White Wins!"
            ts = self.big_font.render(msg, True, C_STATUS)
            self.screen.blit(ts, (sx + 4, WIN_H - 150))
        elif not self.agent_ready:
            ts = self.font.render("Weights unavailable", True, C_RED)
            self.screen.blit(ts, (sx + 4, WIN_H - 140))
        else:
            turn = "Black" if self.game.current_player == 1 else "White"
            ts = self.font.render(f"{turn}'s turn", True, C_STATUS)
            self.screen.blit(ts, (sx + 4, WIN_H - 140))

        # Move count and debug info
        mc = self.sm_font.render(f"Moves: {len(self.game.history)}", True, C_LABEL)
        self.screen.blit(mc, (sx + 4, WIN_H - 70))

        weights_line = self.sm_font.render(self.weights_path.name, True, C_LABEL)
        self.screen.blit(weights_line, (sx + 4, WIN_H - 50))
        status_line = self.sm_font.render(self.agent_error[:24], True, C_LABEL)
        self.screen.blit(status_line, (sx + 4, WIN_H - 30))

        pygame.display.flip()

    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if (not self.game_over
                        and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1):
                    mx, my = ev.pos
                    half = GRID_WIDTH // 2

                    in_board = (BOARD_MARGIN - half <= mx <= BOARD_MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH + half and
                                BOARD_MARGIN - half <= my <= BOARD_MARGIN + (BOARD_SIZE - 1) * GRID_WIDTH + half)
                    if in_board and self.game.current_player == self.player_col:
                        c = round((mx - BOARD_MARGIN) / GRID_WIDTH)
                        r = round((my - BOARD_MARGIN) / GRID_WIDTH)
                        ok, _ = self.game.make_move(r, c)
                        if ok:
                            if self.game.winner:
                                self.game_over = True
                                self.winning_seq = self.game.get_winning_sequence()
                            else:
                                self.draw()
                                pygame.display.flip()
                                time.sleep(0.05)
                                self._do_ai_move()

                for btn in self.btns:
                    btn.handle(ev)

            self.draw()
            self.clock.tick(30)


def parse_args():
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
