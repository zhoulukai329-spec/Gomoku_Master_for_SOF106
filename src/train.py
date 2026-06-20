"""Single-pipeline PPO self-play training for the Gomoku neural agent."""

import argparse

from agent import GomokuAgent
from gomoku_game import GomokuGame
from storage import (
    best_weights_path,
    checkpoint_weights_path,
    create_run_name,
    latest_weights_path,
    resolve_weights_path,
)
from training_logger import TrainingLogger


def check_blocking(game, row, col, player):
    """Return True when a move interrupts an opponent line of length three or more."""
    opponent = 3 - player
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in directions:
        # count opponent stones extending forward from this cell.
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < game.size and 0 <= c < game.size and game.board[r, c] == opponent:
            count += 1
            r += dr
            c += dc
        # count opponent stones extending backward from this cell.
        r, c = row - dr, col - dc
        while 0 <= r < game.size and 0 <= c < game.size and game.board[r, c] == opponent:
            count += 1
            r -= dr
            c -= dc
        # if the opponent had three or more stones lined up around this cell, placing a stone here counts as "blocking" their threat.
        if count >= 3:
            return True
    return False


def max_alignment(board, row, col, player, size):
    """Measure the longest connected line created by a stone."""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    best = 1
    for dr, dc in directions:
        # Count this player's own stones forward and backward in each direction, the same way check_win does, and track the longest one.
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < size and 0 <= c < size and board[r, c] == player:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < size and 0 <= c < size and board[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        best = max(best, count)
    return best


def move_reward(game, row, col, player, blocked_threat):
    """Build a dense reward so early training gets useful feedback."""
    # small bonus for playing closer to the center, since central moves are usually more flexible and this nudges early random play toward more sensible openings.
    center = (game.size - 1) / 2.0
    distance = abs(row - center) + abs(col - center)
    center_bonus = max(0.0, 1.0 - distance / max(center * 2.0, 1.0)) * 0.05

    # reward longer connected lines so the network starts to value building toward a five-in-a-row, not just placing stones at random.
    alignment_bonus = max_alignment(game.board, row, col, player, game.size) * 0.08
    reward = 0.02 + center_bonus + alignment_bonus

    # extra reward for defensive play, i.e. interrupting a dangerous opponent line before it becomes a winning threat.
    if blocked_threat:
        reward += 0.25

    # the dominant signal by far is the actual game outcome —
    # winning gives a large reward, and a draw gives a small positive one
    # so it is still preferred over losing.
    if game.winner == player:
        reward += 5.0
    elif game.winner == 3:
        reward += 0.3

    return reward


def play_match(black_agent, white_agent, size, temperature):
    """Run a deterministic evaluation game between two agents."""
    game = GomokuGame(size)

    # Keep alternating moves between the two agents until the game ends,
    # using deterministic=True so evaluation always reflects each agent's
    # best move rather than its exploratory behavior.
    while game.winner == 0:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        # Pick whichever agent owns the color whose turn it currently is.
        current_agent = black_agent if game.current_player == 1 else white_agent
        move = current_agent.select_action(
            game.board,
            game.current_player,
            legal_moves,
            temperature=temperature,
            deterministic=True,
            record=False,
        )
        game.make_move(*move)

    return game.winner


def evaluate_against_best(agent, args):
    """Compare the current agent against the best saved checkpoint."""
    incumbent_path = best_weights_path(args.run_name)
    if not incumbent_path.exists():
        # on the very first evaluation of a run there is no "best"
        # checkpoint yet, so treat the current agent as automatically
        # passing, which lets it become the first saved "best" checkpoint.
        return 1.0

    # load a separate agent instance for the incumbent so the
    # current agent's own weights are never touched during evaluation.
    incumbent = GomokuAgent(size=args.size, lr=args.lr)
    incumbent.load(incumbent_path)

    # Step play eval_games matches, alternating which side (black or
    # white) the challenger plays, so neither agent gets a free advantage
    # just from always moving first.
    challenger_score = 0.0
    for game_index in range(args.eval_games):
        if game_index % 2 == 0:
            winner = play_match(agent, incumbent, args.size, args.eval_temperature)
            challenger_color = 1
        else:
            winner = play_match(incumbent, agent, args.size, args.eval_temperature)
            challenger_color = 2

        # score 1 point for a challenger win, 0.5 for a draw, and 0 for a loss, which is the standard scoring used to compute a win rate that treats draws fairly.
        if winner == challenger_color:
            challenger_score += 1.0
        elif winner == 3:
            challenger_score += 0.5

    return challenger_score / max(args.eval_games, 1)


def save_training_state(agent, args, episode, save_checkpoint, promote_best):
    """Save latest weights every time and optional checkpoint variants when needed."""
    # the "_latest" file is refreshed on every single episode so
    # training can always be resumed from exactly where it left off.
    latest_path = latest_weights_path(args.run_name)
    agent.save(latest_path)

    # numbered checkpoints are only written occasionally (controlled
    # by --save-every), since saving a full network every episode would be
    # wasteful for long runs.
    checkpoint_path = ""
    if save_checkpoint:
        checkpoint_path = str(checkpoint_weights_path(args.run_name, episode))
        agent.save(checkpoint_path)

    # the "_best" file is only overwritten when this episode's evaluation showed the agent is at least as strong as the previous best, keeping it as a reliable "strongest known policy" checkpoint.
    if promote_best:
        agent.save(best_weights_path(args.run_name))

    return str(latest_path), checkpoint_path


def train(args):
    """Run the full self-play, update, evaluation, and save loop."""
    # settle on one run name up front so every saved file (weights, logs, summary) for this run shares the same prefix and stays grouped.
    run_name = args.run_name or create_run_name("ppo_rl")
    args.run_name = run_name

    # construct the game environment, the trainable agent, and the logger that will record every episode's metrics.
    game = GomokuGame(args.size)
    agent = GomokuAgent(
        size=args.size,
        lr=args.lr,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )
    logger = TrainingLogger(run_name, rolling_window=args.rolling_window)

    # optionally resume from a previously saved checkpoint instead of starting from freshly initialized random weights.
    resume_path = resolve_weights_path(args.weights_path, run_name)
    if args.resume and resume_path.exists():
        agent.load(resume_path)
        print(f"Resumed weights from {resume_path}")

    best_eval_win_rate = 0.0
    last_update_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # the main training loop. Each iteration of this loop is onefull self-play game from an empty board to a win, loss, or draw.
    for episode in range(1, args.episodes + 1):
        game.reset()
        episode_reward = 0.0
        steps = 0

        # play moves until the game reaches a terminal state. 
        while game.winner == 0:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            # use a higher temperature (more random) during the opening to encourage exploring different lines of play, then switch to a lower temperature (more confident/greedy) once the game has progressed past the configured opening length.
            move_number = len(game.history)
            temperature = (
                args.exploration_temperature
                if move_number < args.exploration_moves
                else args.endgame_temperature
            )
            current_player = game.current_player
            blocked_threat = False

            # sample one move from the current policy. record=True tells the agent to store this state/action/logprob in its buffer for the upcoming PPO update.
            move = agent.select_action(
                game.board,
                current_player,
                legal_moves,
                temperature=temperature,
                deterministic=False,
                record=True,
            )
            if move is None:
                break

            row, col = move

            # check whether this move blocked an opponent's threat before the board changes, since check_blocking reads the board state from just before the new stone is placed.
            blocked_threat = check_blocking(game, row, col, current_player)
            ok, error = game.make_move(row, col)
            if not ok:
                # This should never trigger because select_action only ever picks from get_legal_moves(), but it is kept as a safety net in case the masking logic and rule logic ever drift.
                raise RuntimeError(f"Training generated an illegal move: {error}")

            # turn this move's outcome into a numeric reward and record it against the action that was just sampled.
            reward = move_reward(game, row, col, current_player, blocked_threat)
            is_terminal = game.winner != 0
            agent.record_reward(reward, is_terminal)
            episode_reward += reward
            steps += 1

        # run a PPO update once enough episodes have accumulated in the buffer (controlled by --update-every), and always run one final update on the very last episode so no collected data is left unused at the end of training.
        if episode % args.update_every == 0 or episode == args.episodes:
            last_update_stats = agent.update()

        # periodically evaluate the current agent against the best saved checkpoint, and promote it to "best" if it performs at least as well.
        eval_win_rate = 0.0
        promote_best = False
        if episode % args.eval_every == 0 or episode == args.episodes:
            eval_win_rate = evaluate_against_best(agent, args)
            if eval_win_rate >= best_eval_win_rate:
                best_eval_win_rate = eval_win_rate
                promote_best = True

        # always refresh the resumable "latest" weights, and write a numbered checkpoint file on the configured save interval.
        save_checkpoint = episode % args.save_every == 0 or episode == args.episodes
        latest_path, checkpoint_path = save_training_state(
            agent,
            args,
            episode,
            save_checkpoint=save_checkpoint,
            promote_best=promote_best,
        )

        # append this episode's full set of metrics to the CSV log, after saving, so the checkpoint_path recorded in the log row is accurate even when this episode wrote a new checkpoint file.
        rates = logger.log_episode(
            {
                "episode": episode,
                "steps": steps,
                "winner": game.winner if game.winner != 0 else 3,
                "episode_reward": episode_reward,
                "policy_loss": last_update_stats["policy_loss"],
                "value_loss": last_update_stats["value_loss"],
                "entropy": last_update_stats["entropy"],
                "eval_win_rate": eval_win_rate,
                "best_eval_win_rate": best_eval_win_rate,
                "checkpoint_path": checkpoint_path,
            }
        )

        # print a progress line on the configured interval, plus always on the very first and very last episode, so long runs remain easy to monitor without flooding the console.
        if episode % args.log_every == 0 or episode == 1 or episode == args.episodes:
            print(
                f"[Train] Episode {episode:>5}/{args.episodes} "
                f"Reward={episode_reward:>7.3f} Steps={steps:>3} Winner={game.winner or 3} "
                f"PolicyLoss={last_update_stats['policy_loss']:.4f} "
                f"ValueLoss={last_update_stats['value_loss']:.4f} "
                f"Entropy={last_update_stats['entropy']:.4f} "
                f"BlackWin={rates['black_win_rate']:.2%} "
                f"WhiteWin={rates['white_win_rate']:.2%} "
                f"Eval={eval_win_rate:.2%}"
            )

    # once every episode has run, write one final JSON summary capturing where the key output files live and how the run performed.
    logger.write_summary(
        {
            "run_name": run_name,
            "latest_weights": latest_path,
            "best_weights": str(best_weights_path(run_name)),
            "best_eval_win_rate": best_eval_win_rate,
            "episodes": args.episodes,
        }
    )

    print(f"Training finished. Latest weights: {latest_path}")
    print(f"Best weights: {best_weights_path(run_name)}")


def parse_args():
    """Parse command-line options for training and evaluation behavior."""
    parser = argparse.ArgumentParser(description="Train the Gomoku neural PPO agent")
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-clip", dest="eps_clip", type=float, default=0.2)
    parser.add_argument("--k-epochs", dest="k_epochs", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--update-every", type=int, default=8, help="Update every N episodes")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=6)
    parser.add_argument("--eval-temperature", type=float, default=0.05)
    parser.add_argument("--exploration-moves", type=int, default=12)
    parser.add_argument("--exploration-temperature", type=float, default=1.0)
    parser.add_argument("--endgame-temperature", type=float, default=0.2)
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--weights-path", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())