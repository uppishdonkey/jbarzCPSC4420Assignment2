# CPSC 4420 - Assignment 2: Q-Learning with FrozenLake

## Project Description

This project implements a tabular Q-learning reinforcement learning agent using the FrozenLake 8x8 environment from Gymnasium.

The goal is to train an agent to navigate a frozen lake from the starting position to the goal while avoiding holes. The agent learns a policy through trial and error using rewards received from the environment.

This project investigates how different hyperparameter settings affect the learning process and final performance.

## Environment

- **Environment:** FrozenLake-v1
- **Map size:** 8x8
- **State space:** 64 discrete states
- **Action space:** 4 actions (Left, Down, Right, Up)
- **Reward:** +1 for reaching the goal and 0 for other transitions
- **Movement:** Deterministic (`is_slippery=False`)
- **Learning algorithm:** Tabular Q-learning

The Q-table stores an estimated value for each state-action pair.

## Requirements

The project uses Python and the following libraries:

- Gymnasium
- NumPy
- Matplotlib

## Installation

Make sure Python is installed on your computer.

Install the required dependencies by running:

```bash
python -m pip install gymnasium numpy matplotlib
```

## How to Run

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the required dependencies.
4. Run the Python program:

```bash
python q_learning_frozenlake.py
```

The program trains Q-learning agents, performs hyperparameter experiments, evaluates trained policies, and saves the results.

## Q-Learning Algorithm

The agent learns using the following update rule:

Q(s, a) = Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]

where:

- **Q(s, a):** Current estimated value of an action in a state.
- **α (alpha):** Learning rate.
- **γ (gamma):** Discount factor.
- **r:** Reward received after taking an action.
- **s':** Next state.
- **ε (epsilon):** Exploration probability used by the epsilon-greedy policy.

The agent initially explores the environment and gradually reduces its exploration rate during training.

## Experiments

The project includes the following experiments:

### 1. Baseline Training

Trains an agent using the baseline hyperparameters and plots reward versus training episode.

### 2. Learning-Rate Experiment

Compares learning rates:

- 0.1
- 0.5
- 0.9

### 3. Discount-Factor Experiment

Compares discount factors:

- 0.50
- 0.90
- 0.99

### 4. Exploration Experiment

Compares epsilon-decay rates:

- 0.99
- 0.995
- 0.999

### 5. Policy Evaluation

Evaluates trained agents with exploration disabled over 100 episodes. The evaluation reports mean reward, success rate, and mean episode length.

## Results

The program saves generated plots, CSV summaries, and Q-tables in the `results/` directory.

Expected output files include:

- `baseline_rewards.png` — Baseline reward curve.
- `learning_rate_comparison.png` — Learning-rate comparison.
- `discount_factor_comparison.png` — Discount-factor comparison.
- `exploration_comparison.png` — Exploration-schedule comparison.
- `learning_rate_summary.csv` — Learning-rate experiment statistics.
- `discount_factor_summary.csv` — Discount-factor experiment statistics.
- `exploration_summary.csv` — Exploration experiment statistics.
- `evaluation_summary.csv` — Evaluation results for candidate configurations.
- `baseline_q_table.npy` — Baseline Q-table.
- `best_q_table.npy` — Q-table for the selected configuration.

Actual results depend on the training process and hyperparameter settings. Refer to the generated plots and CSV files when discussing experimental findings.

## Repository Structure

```text
.
├── q_learning_frozenlake.py
├── README.md
└── results/
    ├── baseline_rewards.png
    ├── learning_rate_comparison.png
    ├── discount_factor_comparison.png
    ├── exploration_comparison.png
    ├── learning_rate_summary.csv
    ├── discount_factor_summary.csv
    ├── exploration_summary.csv
    ├── evaluation_summary.csv
    ├── baseline_q_table.npy
    └── best_q_table.npy
```

The `results/` directory is generated when the program runs. The exact files available depend on whether the experiments have completed successfully.

## Conclusion

This project demonstrates how tabular Q-learning can be applied to a discrete reinforcement learning environment. By comparing learning rates, discount factors, and exploration schedules, the experiments investigate how hyperparameters influence training behavior and policy performance.

The evaluation results provide a basis for selecting and discussing the best-performing configuration.
