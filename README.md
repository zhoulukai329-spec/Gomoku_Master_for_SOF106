# Gomoku Master for SOF106

本项目实现了一个基于强化学习的五子棋 AI 系统。当前代码库采用的是基于 `PPO` 的 actor-critic 智能体，配合自博弈训练流程与 `Pygame` 图形界面，支持模型训练、人机对弈和训练产物保存。

## 1. 本地部署、训练与开始游戏

### 1.1 环境要求

- 推荐操作系统：Windows
- 推荐 Python 版本：`Python 3.10+`
- 主要依赖：
  - `numpy>=1.26`
  - `torch>=2.2`
  - `pygame>=2.5`

### 1.2 克隆项目

```bash
git clone <your-repository-url>
cd Gomoku_Master_for_SOF106
```

### 1.3 创建虚拟环境

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 1.4 安装依赖

```bash
pip install -r requirements.txt
```

如果你的环境中 `torch` 没有成功安装，请根据本机的 CPU 或 CUDA 配置，从 PyTorch 官网选择对应版本安装：

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

### 1.5 项目结构

```text
Gomoku_Master_for_SOF106/
├─ README.md
├─ requirements.txt
└─ src/
   ├─ agent.py
   ├─ gomoku_game.py
   ├─ gui.py
   ├─ model.py
   ├─ storage.py
   ├─ train.py
   └─ training_logger.py
```

### 1.6 如何在本地训练

训练入口为 `src/train.py`。由于训练脚本直接导入同级模块，因此建议在 `src` 目录下执行。

```bash
cd src
python train.py
```

常用训练命令示例：

```bash
python train.py --episodes 2000 --size 15
python train.py --episodes 500 --run-name exp01
python train.py --resume --run-name exp01
python train.py --resume --run-name exp01 --weights-path artifacts\weights\exp01_latest.pth
```

常用参数说明：

- `--episodes`：自博弈训练总局数
- `--size`：棋盘大小，默认值为 `15`
- `--run-name`：本次训练的自定义名称
- `--resume`：从已有检查点继续训练
- `--weights-path`：显式指定权重文件路径

### 1.7 训练输出内容

训练启动后，项目会自动在 `src/artifacts/` 下生成相关产物：

- `src/artifacts/weights/`：模型权重文件
- `src/artifacts/logs/`：训练过程日志与总结文件

典型文件包括：

- `artifacts\weights\ppo_rl_latest.pth`：默认运行名下的最新权重
- `artifacts\weights\ppo_rl_best.pth`：评估效果最好的权重
- `artifacts\logs\<run-name>_training.csv`：逐局训练指标日志
- `artifacts\logs\<run-name>_summary.json`：一次训练完成后的总结信息

### 1.8 训练完成后如何开始游戏

游戏图形界面入口为 `src/gui.py`，同样建议在 `src` 目录下执行。

如果你使用默认运行名完成训练：

```bash
cd src
python gui.py
```

如果你希望加载指定权重：

```bash
cd src
python gui.py --weights-path artifacts\weights\exp01_latest.pth
```

说明：

- 如果未传入 `--weights-path`，GUI 会尝试加载默认的最新权重。
- 如果没有找到有效权重文件，GUI 可能仍然可以打开，但 AI 不一定能够正常工作。
- 推荐的本地流程如下：
  1. 创建并激活虚拟环境
  2. 安装依赖
  3. 运行训练脚本
  4. 确认权重文件已生成
  5. 启动 GUI，与训练后的 AI 对战

## 2. 项目简短报告

### a) Title

**PPO-Based Gomoku AI with Self-Play Training and Pygame Interface**

### b) Abstract

本项目开发了一个能够通过强化学习进行训练，并可用于人机对弈的五子棋智能系统。系统由五子棋规则引擎、策略价值神经网络、基于 PPO 的训练智能体以及 Pygame 图形界面组成。项目最终形成了一条完整流程，支持自博弈训练、模型检查点保存、与历史最佳模型对抗评估，以及训练完成后的交互式游戏体验。

### c) Introduction

我们选择五子棋作为本项目的研究主题，是因为它规则清晰、策略空间丰富，同时适合作为强化学习算法的实验对象。对本专业和研究方向而言，五子棋能够很好地连接人工智能中的多个核心问题，包括棋盘状态表示、动作决策、神经网络建模以及智能体与环境交互式学习。

本项目的意义在于，它不仅实现了一个可训练的游戏 AI，也展示了如何将算法模块、规则模块、日志模块和图形界面模块整合为一个完整的软件系统。这使得项目同时具有算法实践价值和工程实现价值。

### d) Methodology

项目采用模块化的方法进行设计与实现，主要流程如下：

1. 实现五子棋规则引擎，用于维护棋盘状态、合法落子、胜负判定、历史记录以及黑棋禁手规则。
2. 构建神经网络模型，对棋盘状态进行编码，并输出策略分布与局面价值。
3. 实现基于 PPO 的强化学习智能体，用于采样动作、记录轨迹、更新参数与保存模型。
4. 构建自博弈训练流程，使智能体通过不断对弈持续优化策略。
5. 构建 Pygame 图形界面，使用户能够在本地与训练后的模型进行对弈。
6. 通过日志与总结文件记录训练过程，便于后续分析模型表现。

在信息识别、筛选、处理和分析方面，我们主要基于仓库中的实际实现模块进行整理。当前项目结构将规则逻辑、模型逻辑、训练逻辑、存储日志逻辑和界面逻辑分离，有助于降低模块耦合度并提升可维护性。

团队分工可以按模块职责描述如下：

- 游戏规则与棋盘逻辑：`gomoku_game.py`
- 神经网络模型与输入编码：`model.py`
- 强化学习智能体：`agent.py`
- 训练与评估流程：`train.py`
- 图形界面与人机交互：`gui.py`
- 训练产物路径管理与日志记录：`storage.py` 和 `training_logger.py`

如果你们后续需要提交正式课程报告，可以将上述“按模块职责分工”的内容替换为实际成员姓名及其对应任务。

### e) Validation/Verification

项目主要通过以下方式对结果进行验证：

1. 在训练过程中，当前模型会周期性地与历史最佳模型进行对抗评估，用于验证训练是否带来了性能提升。
2. 训练过程中的奖励、策略损失、价值损失、熵、胜率以及评估胜率等指标会被写入 CSV 日志文件，便于观察训练趋势。
3. 项目会保存 latest 和 best 两类权重文件，便于断点续训、结果对比与回归验证。
4. 训练完成后，可以通过图形界面进行人机对弈，从实际使用角度验证模型是否具备基本的对局能力。

当前仓库中尚未包含独立的自动化测试套件，因此验证方式主要依赖训练期评估、日志分析以及人机对战体验。不过，这种“自博弈评估 + 日志记录 + 交互式验证”的组合，仍能为模型效果提供较直接的证据。

### f) Conclusion

总体而言，本项目完成了一个较完整的五子棋 AI 工作流，涵盖了规则模拟、PPO 自博弈训练、模型评估、训练产物管理以及图形化人机对弈界面。项目的主要成果不仅是得到一个可训练的 AI 模型，更重要的是搭建了一个从训练到使用都可落地运行的完整系统。

从工程角度看，项目体现了模块化设计的重要性。规则、学习算法、训练流程和界面逻辑被拆分到不同文件中，结构相对清晰，便于维护和后续扩展。通过本项目，我们进一步积累了在强化学习、神经网络建模、训练验证以及 AI 应用集成方面的实践经验。

未来可以继续改进的方向包括：

- 增加自动化单元测试与集成测试
- 引入更强的评估基线与更系统的对照实验
- 支持更多可配置的棋盘大小与游戏模式
- 优化 GUI 交互体验与对局反馈信息
- 在训练流程中加入更强的搜索策略或混合决策方法

