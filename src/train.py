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
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < game.size and 0 <= c < game.size and game.board[r, c] == opponent:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < game.size and 0 <= c < game.size and game.board[r, c] == opponent:
            count += 1
            r -= dr
            c -= dc
        if count >= 3:
            return True
    return False


def max_alignment(board, row, col, player, size):
    """Measure the longest connected line created by a stone."""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    best = 1
    for dr, dc in directions:
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
    # Favor moves close to the center because they are usually more flexible.
    center = (game.size - 1) / 2.0
    distance = abs(row - center) + abs(col - center)
    center_bonus = max(0.0, 1.0 - distance / max(center * 2.0, 1.0)) * 0.05

    # Reward longer local connections so the agent learns shape building.
    alignment_bonus = max_alignment(game.board, row, col, player, game.size) * 0.08
    reward = 0.02 + center_bonus + alignment_bonus

    # Extra reward is given when the move breaks an opponent threat.
    if blocked_threat:
        reward += 0.25

    # Winning gets a large bonus. Draws get a small neutral reward.
    if game.winner == player:
        reward += 5.0
    elif game.winner == 3:
        reward += 0.3

    return reward


def play_match(black_agent, white_agent, size, temperature):
    """Run a deterministic evaluation game between two agents."""
    game = GomokuGame(size)

    while game.winner == 0:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            break

        # Select the correct agent based on whose turn it is.
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
        # If no best checkpoint exists yet, treat the current model as the baseline.
        return 1.0

    incumbent = GomokuAgent(size=args.size, lr=args.lr)
    incumbent.load(incumbent_path)

    # Alternate colors so neither side gets a fixed first-move advantage.
    challenger_score = 0.0
    for game_index in range(args.eval_games):
        if game_index % 2 == 0:
            winner = play_match(agent, incumbent, args.size, args.eval_temperature)
            challenger_color = 1
        else:
            winner = play_match(incumbent, agent, args.size, args.eval_temperature)
            challenger_color = 2

        if winner == challenger_color:
            challenger_score += 1.0
        elif winner == 3:
            challenger_score += 0.5

    return challenger_score / max(args.eval_games, 1)


def save_training_state(agent, args, episode, save_checkpoint, promote_best):
    """Save latest weights every time and optional checkpoint variants when needed."""
    latest_path = latest_weights_path(args.run_name)
    agent.save(latest_path)

    checkpoint_path = ""
    if save_checkpoint:
        checkpoint_path = str(checkpoint_weights_path(args.run_name, episode))
        agent.save(checkpoint_path)

    if promote_best:
        agent.save(best_weights_path(args.run_name))

    return str(latest_path), checkpoint_path


def train(args):
    """Run the full self-play, update, evaluation, and save loop."""
    # Create a readable run name once so every artifact uses the same prefix.
    run_name = args.run_name or create_run_name("ppo_rl")
    args.run_name = run_name

    # Build the game environment, agent, and logger for this run.
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

    # Resume from an older checkpoint when the caller asks for it.
    resume_path = resolve_weights_path(args.weights_path, run_name)
    if args.resume and resume_path.exists():
        agent.load(resume_path)
        print(f"Resumed weights from {resume_path}")

    best_eval_win_rate = 0.0
    last_update_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # Each episode is one full self-play game.
    for episode in range(1, args.episodes + 1):
        game.reset()
        episode_reward = 0.0
        steps = 0

        # Keep playing until the game reaches a win or draw state.
        while game.winner == 0:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            # Explore more in the opening and become greedier later in the game.
            move_number = len(game.history)
            temperature = (
                args.exploration_temperature
                if move_number < args.exploration_moves
                else args.endgame_temperature
            )
            current_player = game.current_player
            blocked_threat = False

            # Sample one move and store the rollout data for PPO.
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

            # Measure whether the move blocked danger before the board changes.
            blocked_threat = check_blocking(game, row, col, current_player)
            ok, error = game.make_move(row, col)
            if not ok:
                raise RuntimeError(f"Training generated an illegal move: {error}")

            # Convert the move outcome into the training reward signal.
            reward = move_reward(game, row, col, current_player, blocked_threat)
            is_terminal = game.winner != 0
            agent.record_reward(reward, is_terminal)
            episode_reward += reward
            steps += 1

        # Update the network after enough episodes have been collected.
        if episode % args.update_every == 0 or episode == args.episodes:
            last_update_stats = agent.update()

        # Periodically compare the new policy against the best saved one.
        eval_win_rate = 0.0
        promote_best = False
        if episode % args.eval_every == 0 or episode == args.episodes:
            eval_win_rate = evaluate_against_best(agent, args)
            if eval_win_rate >= best_eval_win_rate:
                best_eval_win_rate = eval_win_rate
                promote_best = True

        # Always refresh latest weights and sometimes write extra checkpoints.
        save_checkpoint = episode % args.save_every == 0 or episode == args.episodes
        latest_path, checkpoint_path = save_training_state(
            agent,
            args,
            episode,
            save_checkpoint=save_checkpoint,
            promote_best=promote_best,
        )

        # Append the episode to the CSV log after the save step is done.
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

        # Print periodic progress so long runs remain easy to monitor.
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

    # Write one compact summary file at the very end of the run.
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
