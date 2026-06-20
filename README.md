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

## 2. Short Project Report

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
