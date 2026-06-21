# Gomoku Master for SOF106

This project implements a Gomoku AI system based on reinforcement learning. The current codebase uses a `PPO`-based actor-critic agent together with a self-play training pipeline and a `Pygame` GUI, supporting model training, human-vs-AI gameplay, and artifact persistence.

## 1. Local Setup, Training, and Gameplay

### 1.1 Environment Requirements

- Recommended operating system: Windows
- Recommended Python version: `Python 3.10+`
- Main dependencies:
  - `numpy>=1.26`
  - `torch>=2.2`
  - `pygame>=2.5`

### 1.2 Clone the Project

```bash
git clone https://github.com/zhoulukai329-spec/Gomoku_Master_for_SOF106.git
cd Gomoku_Master_for_SOF106
```

### 1.3 Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate # use venv

conda activate <env-name> # use conda env

python -m pip install --upgrade pip
```

### 1.4 Install Dependencies

```bash
pip install -r requirements.txt
```

If `torch` is not installed successfully in your environment, install the appropriate PyTorch build for your CPU or CUDA setup from the official website:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

### 1.5 Project Structure

```text
Gomoku_Master_for_SOF106/
├─ README.md
├─ requirements.txt
└─ src/
   ├─ artifacts/
   │   └─ weights/ # model weight files(not uploaded online)
   │   └─ logs/ # training logs and summary files(not uploaded online)
   ├─ agent.py
   ├─ gomoku_game.py
   ├─ gui.py
   ├─ model.py
   ├─ storage.py
   ├─ train.py
   └─ training_logger.py
```

### 1.6 How to Train Locally

The training entry point is `src/train.py`. Since the training script imports sibling modules directly, it is recommended to run it from the `src` directory.

Common training commands:

```bash
cd src
python train.py --episodes 2000 --size 15
python train.py --episodes 500 --run-name train01
python train.py --resume --run-name train01
python train.py --resume --run-name train01 --weights-path artifacts\weights\train01_latest.pth
```

Common arguments:

- `--episodes`: total number of self-play episodes
- `--size`: board size, default is `15`
- `--run-name`: custom name for the current training run
- `--resume`: continue training from an existing checkpoint
- `--weights-path`: explicitly specify the weights file path

### 1.7 Training Outputs

After training starts, the project automatically generates artifacts under `src/artifacts/`:

- `src/artifacts/weights/`: model weight files
- `src/artifacts/logs/`: training logs and summary files

Typical files include:

- `artifacts\weights\ppo_rl_latest.pth`: the latest weights for the default run name
- `artifacts\weights\ppo_rl_best.pth`: the best-performing weights according to evaluation
- `artifacts\logs\<run-name>_training.csv`: per-episode training metrics
- `artifacts\logs\<run-name>_summary.json`: summary information after one training run finishes

### 1.8 How to Start the Game After Training

The graphical gameplay entry point is `src/gui.py`, and it is also recommended to run it from the `src` directory.

```bash
cd src
python gui.py --weights-path <path to the weight file>
```

Notes:

- If `--weights-path` is not provided, the GUI will try to load the default **latest weights**.
- If no valid weights file is found, the GUI may still open, but the AI may not function correctly.
- A recommended local workflow is:
  1. create and activate a virtual environment
  2. install dependencies
  3. run the training script
  4. confirm that weight files have been generated
  5. launch the GUI and play against the trained AI

## 2. AI Methods Used in This Project

This section explains the full AI stack used in the project. It starts from the big picture, then goes into the core methods one by one. The goal is to stay technically correct while still being easy to follow for readers who are new to AI.

### 2.1 AI Method Map by Category

The project does **not** use every common AI category. It mainly uses reinforcement learning and deep learning, plus some rule-based game logic that acts as the environment and safety layer.

#### A. Reinforcement Learning Methods

- `PPO (Proximal Policy Optimization)`: the main learning algorithm
- `Actor-Critic`: the learning structure used inside PPO
- `Self-Play`: the training setting where the agent learns by playing against itself
- `Reward Shaping`: extra reward design that gives the agent more learning signals than only win or loss

#### B. Deep Learning Methods

- `Convolutional Neural Network (CNN)`: used to read the board as a spatial pattern
- `Residual Network (ResNet-style blocks)`: used to make the network deeper and more stable to train
- `Shared Encoder + Policy Head + Value Head`: one network produces both move preference and position quality

#### C. Decision Control and Inference Methods

- `Legal Action Masking`: blocks illegal moves before the model chooses an action
- `Temperature Scaling`: controls how random or sharp the move choice is
- `Deterministic Inference`: used in evaluation and GUI play for stronger and more stable moves
- `Tactical Move Heuristic`: masks urgent one-move wins and blocks before trusting raw policy scores

#### D. Optimization and Stability Methods

- `Adam`: the optimizer used to update neural network weights
- `Return Normalization`: keeps training targets on a stable scale
- `Entropy Bonus`: keeps exploration alive
- `Gradient Clipping`: avoids unstable weight jumps
- `Best-Checkpoint Evaluation`: compares the current model with the best saved model

#### E. Rule-Based Game Intelligence

- `Gomoku / Renju Rule Engine`: checks legal moves, win conditions, forbidden moves, and game history

#### F. Methods Not Used in This Project

To keep the technical story accurate, it is also important to say what is **not** used:

- No supervised learning dataset training
- No generative AI model
- No transformer model
- No Monte Carlo Tree Search (MCTS)
- No full minimax search with handcrafted game-tree expansion

This means the neural policy mainly learns from repeated play experience, while a small tactical layer handles immediate wins and blocks during deterministic play.

### 2.2 How the Whole AI Pipeline Works

The full pipeline can be understood as the following loop:

```text
Board state
-> State encoding
-> Policy-value neural network
-> Legal move mask
-> Tactical move mask for urgent wins and blocks
-> Move selection
-> Game engine applies move
-> Reward is computed
-> Experience is stored
-> PPO updates the network
-> Best/latest weights are saved
-> GUI loads weights for human-vs-AI play
```

An easy way to understand this is to imagine a student learning chess without a teacher:

- the `rule engine` plays the role of the referee
- the `neural network` plays the role of intuition
- the `reward function` plays the role of feedback
- `PPO` plays the role of the training coach that says "improve, but do not change too much at once"

### 2.3 Reinforcement Learning Core: PPO

`PPO` is the core algorithm of the project. It is the main method that turns game experience into better model weights.

#### 2.3.1 What PPO Tries to Do

The goal of PPO is simple:

- improve the policy from experience
- keep learning stable
- avoid updates that are too large and break previous good behavior

In plain words, PPO tries to teach the agent like a careful coach. If a new idea looks better, PPO accepts it, but it does not allow the model to suddenly become a completely different player after one update.

#### 2.3.2 Why Not Use a Simpler Policy Gradient

Traditional policy gradient methods can be unstable because:

- one update may push the policy too far
- good actions found in earlier steps may be forgotten
- training can become noisy when reward is delayed

PPO addresses this by clipping the policy update. This is the main reason it is widely used in reinforcement learning practice.

#### 2.3.3 PPO in One Simple Formula

The key PPO idea is:

```text
Use the smaller one between:
1. the normal policy improvement term
2. the clipped policy improvement term
```

In more standard notation:

```text
L = min(r_t * A_t, clip(r_t, 1-epsilon, 1+epsilon) * A_t)
```

Where:

- `r_t` is the probability ratio between the new policy and the old policy
- `A_t` is the advantage, meaning "was this action better or worse than expected?"
- `epsilon` is the safe update range

A simple life analogy:

- imagine a student scored a bit better after trying a new study method
- a normal method may say "great, change everything now"
- PPO says "good, but only change within a safe range until we are sure"

That safe range is the clipping idea.

#### 2.3.4 PPO Components Used in This Project

The project uses the following PPO elements:

- `old policy` and `current policy`
- discounted returns
- advantage estimation in a simple form: `return - value`
- clipped surrogate objective
- value loss with `MSE`
- entropy bonus for exploration
- multiple update passes over one rollout

These parts are implemented in `agent.py`.

#### 2.3.5 PPO Step-by-Step in This Project

The training logic can be broken into the following steps:

1. The agent observes the current board.
2. The board is turned into a 3-channel tensor.
3. The old frozen policy outputs move logits and a state value.
4. Illegal moves are masked out.
5. Forced tactical wins and blocks are masked first, then a move is sampled during training.
6. The game engine applies the move.
7. A reward is computed from the result of the move.
8. State, action, log-probability, reward, and terminal flag are stored.
9. After enough episodes, PPO computes discounted returns.
10. PPO compares the current policy with the old policy.
11. PPO updates network weights with clipping, value loss, and entropy bonus.
12. The updated policy is copied into the old policy for the next rollout.

The text flow below shows the same idea:

```text
Collect game experience
-> Store rollout
-> Compute returns
-> Compute advantage
-> Compute PPO clipped loss
-> Update network
-> Sync old policy
-> Start next round
```

#### 2.3.6 How the Main PPO Terms Should Be Read

- `return`: total future reward after one move
- `value`: the critic's guess of how good the position is
- `advantage = return - value`: how much better the real outcome was than expected
- `entropy`: how spread out the move probabilities are
- `clip`: a safety belt for policy change

A simple example:

- if the model thinks a move is average, but the final result becomes very good, the `advantage` is positive
- PPO then increases the chance of that move
- but clipping stops the increase from becoming too large in one jump

#### 2.3.7 PPO Hyperparameters Used Here

The current training script uses these default settings:

- `lr = 3e-4`
- `gamma = 0.99`
- `eps_clip = 0.2`
- `k_epochs = 4`
- `entropy_coef = 0.02`
- `value_coef = 0.5`
- `max_grad_norm = 1.0`
- `update_every = 8`
- `eval_every = 50`
- `eval_games = 6`
- `exploration_moves = 12`
- `exploration_temperature = 1.0`
- `endgame_temperature = 0.2`

These values are a practical balance between learning speed and stability:

- `gamma = 0.99` keeps long-term reward important, which matters in board games
- `eps_clip = 0.2` allows policy improvement but still limits large jumps
- `k_epochs = 4` lets the model learn more from each collected batch
- `entropy_coef = 0.02` keeps some exploration alive
- `value_coef = 0.5` gives the critic enough weight without letting it dominate
- `max_grad_norm = 1.0` reduces the risk of unstable updates
- the temperature schedule makes the opening more exploratory and the late game more focused

#### 2.3.8 How PPO Is Adapted to Gomoku in This Project

PPO is a general reinforcement learning method, but this project adapts it to Gomoku in several specific ways:

- the action space is a `15 x 15` board, so one move is one board cell
- illegal cells are blocked by masking before sampling
- the same network can play both black and white because the input is built from the current player's view
- the reward is not only final win/loss reward; it also includes local shape feedback
- evaluation uses low-temperature deterministic play to test real strength more clearly

This adaptation is important. Without legal-move masking, current-player encoding, and reward shaping, PPO would learn much more slowly.

#### 2.3.9 PPO Strengths and Limits in This Project

Strengths:

- stable compared with plain policy gradient
- works directly from self-play experience
- does not need labeled expert data
- fits well with a policy-value network

Limits:

- training can still be slow for a game as large as Gomoku
- without search, tactical depth is limited
- reward design strongly affects learning quality
- evaluation with only a few games can be noisy

This is an important design note for the project: the current AI is mainly a learning-based player, not a search-heavy player.

### 2.4 Actor-Critic Structure

`Actor-Critic` is not a separate training loop here; it is the model structure used by PPO.

#### 2.4.1 What the Actor Does

The `actor` answers:

- "Which move should I play?"

It outputs one score for each board cell. After masking illegal moves and applying softmax, those scores become move probabilities.

#### 2.4.2 What the Critic Does

The `critic` answers:

- "How good is this board position for the current player?"

It outputs one scalar value in the range `[-1, 1]`.

#### 2.4.3 Why Use Both Together

Using only a policy is like choosing moves without a judge.

Using actor and critic together is like:

- one part chooses actions
- one part gives a quality check on the board state

This makes learning more data-efficient and more stable.

### 2.5 Deep Learning Core: CNN with Residual Blocks

The project uses a convolutional neural network with residual blocks as the backbone of the AI model.

#### 2.5.1 Why a CNN Fits Gomoku

Gomoku is a board game, so local spatial patterns matter:

- two connected stones
- open three
- broken four
- blocking lines

A CNN is a natural fit because it reads nearby cells together, just like image models read nearby pixels.

An easy analogy:

- if the board is like a map
- a CNN is like a scanner that looks at small local areas first, then combines them into a larger understanding

#### 2.5.2 Why Residual Blocks Are Used

Residual blocks add skip connections. Instead of forcing each layer to build everything from zero, the network can keep useful old features and learn only the extra change.

This helps because:

- deeper networks are easier to train
- gradients pass more safely through the network
- local board shapes can be combined into richer patterns

#### 2.5.3 Network Structure in This Project

The model uses:

- a `3-channel` board input
- one shared feature extractor
- `5` residual blocks by default
- one `policy head`
- one `value head`

The policy head outputs `size * size` logits, one for each cell.

The value head outputs one scalar for state evaluation.

### 2.6 State Representation: How the Board Becomes Model Input

The board is not fed into the model as raw integers only. It is converted into three planes:

- channel 0: current player's stones
- channel 1: opponent's stones
- channel 2: empty cells

This is a strong design choice because it keeps the meaning of the input consistent:

- the network always sees the board from "my side"
- it does not need two totally different policies for black and white

Simple example:

```text
Channel 0 = my stones
Channel 1 = enemy stones
Channel 2 = empty places
```

This is like giving the model three transparent map layers stacked together.

### 2.7 Self-Play Learning

`Self-play` means the model improves by playing against itself.

#### Why Self-Play Matters

In many board games, it is expensive to collect a large expert dataset. Self-play avoids that need.

The idea is:

- the current model generates its own experience
- that experience becomes training data
- the model gradually becomes a stronger version of itself

This is similar to a player practicing by replaying many games against a clone at slightly different skill states.

#### How It Works Here

- each episode is one full game
- both sides are controlled by the same learning system
- the model stores the rollout
- after a fixed number of episodes, PPO updates the weights

#### Limitation

Self-play is powerful, but if the player is weak at the start, it may spend a long time learning weak habits. This is one reason why reward shaping and evaluation design matter a lot in this project.

### 2.8 Reward Shaping

The project does not wait only for the final win or loss. It adds extra rewards so the model can learn faster.

#### Reward Signals Used Here

The reward function includes:

- a small base reward
- a `center bonus`, which prefers flexible central moves
- an `alignment bonus`, which prefers longer local connections
- a `blocking bonus`, when a move stops an opponent threat
- a stronger terminal reward for actually finishing the game
- a small neutral reward for a draw

The alignment reward is intentionally smaller than the terminal reward, so the agent is encouraged to complete a win instead of only building attractive four-in-a-row shapes.

#### Why Reward Shaping Is Needed

If the model only gets reward at the end of a long game, learning is very slow. Reward shaping gives earlier hints.

Life analogy:

- final win/loss reward is like only getting exam results at the end of the year
- reward shaping is like getting homework feedback every week

#### Risk of Reward Shaping

Reward shaping must be designed carefully. If the bonus is too strong in the wrong place, the model may learn shortcuts that look good to the reward function but do not lead to strong play.

### 2.9 Legal Action Masking

The network outputs one score for every board cell, but not every cell is legal.

So the project uses `legal action masking`:

- illegal moves get a very large negative score
- after softmax, their probability becomes almost zero

This means the model is forced to choose only legal moves.

This is an important bridge between the learning model and the game rules. It keeps UI logic and rule logic separate from the neural policy while still enforcing correct play.

### 2.10 Temperature Scaling and Move Selection

The project uses `temperature` to control how random the move choice is.

- high temperature -> flatter probability -> more exploration
- low temperature -> sharper probability -> more greedy play

#### Training Mode

During training:

- the opening uses more exploration
- the endgame uses a lower temperature
- immediate winning moves and required blocks are treated as forced tactical actions

This gives a balance between trying new ideas and finishing games more decisively.

#### Evaluation and GUI Mode

During evaluation and GUI play:

- the project uses deterministic inference
- the GUI also uses a very low temperature (`0.05`)
- a tactical layer first checks whether the AI can win immediately or must block an opponent's immediate win

This is like telling the model:

- "during practice, try different moves"
- "during the real test, play your best move"

### 2.11 Optimization and Stability Methods

These methods are not the main AI idea, but they are important for making training work in practice.

#### Adam Optimizer

`Adam` updates the neural network weights. It is widely used because it adapts learning rates and usually converges faster than plain gradient descent.

#### Return Normalization

Returns are normalized to a stable scale before value learning. This makes the critic target less noisy.

#### Entropy Bonus

Entropy bonus keeps the move distribution from collapsing too early into one narrow behavior. In simple words, it keeps the agent curious for longer.

#### Gradient Clipping

Gradient clipping prevents one bad batch from causing a huge weight jump. It acts like a safety brake.

### 2.12 Best-Model Evaluation

The project does not only save the latest model. It also compares the current model with the best saved model.

The logic is:

- train for some episodes
- run evaluation games
- if the current model is at least as good as the best saved model, update the best checkpoint

This gives a practical answer to the question:

- "Is the new model really better, or just different?"

### 2.13 Rule Engine as the Environment Layer

Strictly speaking, the Gomoku rule engine is not a learning algorithm. However, it is a key part of the AI system because reinforcement learning needs an environment.

This rule engine provides:

- board state update
- legal move checking
- win detection
- draw handling
- move history
- Renju-style forbidden move checks for black, such as overline, double-four, and double-three

Without this layer, the AI would not have a valid world to learn from.

### 2.14 Full End-to-End Logic in the Project

The end-to-end AI logic of the project can be summarized as:

```text
Rules define the environment
-> Board state is encoded into 3 channels
-> Residual actor-critic network reads the state
-> Policy head suggests moves
-> Value head estimates board quality
-> Legal masking removes invalid moves
-> Temperature controls exploration or greedy play
-> Reward shaping gives learning signals
-> PPO updates the network safely
-> Best/latest checkpoints are stored
-> GUI loads the trained weights for play
```

### 2.15 Practical Value of the AI Stack

For technical readers, this project shows a complete learning pipeline:

- environment design
- state representation
- model design
- policy optimization
- evaluation
- deployment into a usable interface

For non-technical readers, the value is also clear:

- the AI does not rely on fixed scripts only
- it improves through repeated practice
- the system is able to turn training into a playable product

### 2.16 Current Limits and Future Algorithm Directions

To keep the explanation honest, the current algorithm design also has limits:

- it does not use expert demonstrations
- it does not use MCTS or deep search
- it only uses a shallow tactical heuristic for immediate wins and blocks
- it depends strongly on reward design
- self-play may need many episodes before strong strategy appears
- short evaluation matches can make progress look noisy

A practical future direction is to keep the current modular split and strengthen only the learning layer. For example:

- improve reward shaping with better tactical signals
- increase evaluation depth
- mix self-play with stronger fixed opponents
- add a search layer above the policy network

This keeps UI logic and game-rule logic out of the learning core as much as possible, and it follows the idea of low coupling and high cohesion in the project architecture.

## 3. Short Project Report

### a) Title

**PPO-Based Gomoku AI with Self-Play Training and Pygame Interface**

### b) Abstract

This project develops an intelligent Gomoku system that can be trained through reinforcement learning and then used for human-vs-AI gameplay. The system consists of a Gomoku rule engine, a policy-value neural network, a PPO-based training agent, and a Pygame graphical interface. The final outcome is a complete pipeline that supports self-play training, model checkpoint saving, evaluation against the historically best model, and interactive gameplay after training.

### c) Introduction

We selected Gomoku as the topic of this project because it has clear rules, a rich strategic space, and is well suited as an experimental task for reinforcement learning algorithms. In terms of our field of study and interests, Gomoku connects several core topics in artificial intelligence, including board-state representation, action selection, neural network modeling, and learning through repeated interaction between an agent and its environment.

The significance of this project lies in the fact that it not only implements a trainable game AI, but also demonstrates how algorithm modules, rule modules, logging modules, and GUI modules can be integrated into a complete software system. This gives the project both algorithmic value and engineering value.

### d) Methodology

The project is designed and implemented in a modular way, with the main workflow as follows:

1. Implement a Gomoku rule engine to maintain board state, legal moves, win/loss checking, move history, and black forbidden-move rules.
2. Build a neural network model that encodes the board state and outputs a policy distribution and state value.
3. Implement a PPO-based reinforcement learning agent for action sampling, trajectory recording, parameter updates, and model saving.
4. Build a self-play training loop so the agent can continuously improve through repeated gameplay.
5. Build a Pygame graphical interface so users can play against the trained model locally.
6. Record the training process through logs and summary files for later analysis.

For information identification, selection, processing, and analysis, we mainly organized the report based on the actual implemented modules in the repository. The current project structure separates rule logic, model logic, training logic, storage/logging logic, and interface logic, which helps reduce coupling and improve maintainability.

Team responsibilities can be described by module ownership as follows:

- Game rules and board logic: `gomoku_game.py`
- Neural network model and input encoding: `model.py`
- Reinforcement learning agent: `agent.py`
- Training and evaluation pipeline: `train.py`
- Graphical interface and human-computer interaction: `gui.py`
- Artifact path management and logging: `storage.py` and `training_logger.py`

If you later need a formal course submission, the module-based division above can be replaced with actual team member names and their assigned tasks.

### e) Validation/Verification

The project validates its results mainly in the following ways:

1. During training, the current model is periodically evaluated against the historically best model to verify whether training improves performance.
2. Metrics such as reward, policy loss, value loss, entropy, win rate, and evaluation win rate are written to CSV logs so training trends can be observed.
3. The project saves both `latest` and `best` weights, which supports resumed training, result comparison, and regression checking.
4. After training, users can play against the model through the graphical interface to validate whether it has acquired basic gameplay ability from a practical perspective.

The current repository does not yet include a standalone automated test suite, so validation mainly relies on in-training evaluation, log analysis, and human-vs-AI gameplay experience. Even so, this combination of self-play evaluation, log recording, and interactive verification still provides direct evidence of model performance.

### f) Conclusion

Overall, this project completes a relatively full Gomoku AI workflow, covering rule simulation, PPO self-play training, model evaluation, training artifact management, and a graphical human-vs-AI interface. The main achievement is not only obtaining a trainable AI model, but also building a complete system that can run from training to actual use.

From an engineering perspective, the project highlights the importance of modular design. Rules, learning algorithms, training flow, and interface logic are separated into different files, making the structure relatively clear and easier to maintain and extend. Through this project, we further gained practical experience in reinforcement learning, neural network modeling, training validation, and AI application integration.

Possible future improvements include:

- adding automated unit tests and integration tests
- introducing stronger evaluation baselines and more systematic comparison experiments
- supporting more configurable board sizes and game modes
- improving GUI interaction and in-game feedback
- adding stronger search strategies or hybrid decision methods into the training pipeline
