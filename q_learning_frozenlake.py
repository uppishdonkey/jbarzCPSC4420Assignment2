import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# CPSC 4420 - Artificial Intelligence Assignment 2
# Tabular Q-Learning with FrozenLake 8x8
# Undergraduate version: Parts A-G

ENV_NAME = "FrozenLake-v1"
MAP_NAME = "8x8"

# Baseline hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
EPISODES = 20000
MAX_STEPS = 200

EVALUATION_EPISODES = 100
SMOOTHING_WINDOW = 200

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# PART A: Environment description
# FrozenLake is a grid-world environment. The agent starts at
# the top-left cell and tries to reach the goal at the
# bottom-right cell without falling into holes.
# State space: 64 discrete grid cells.
# Action space: 4 actions:
#   0 = Left, 1 = Down, 2 = Right, 3 = Up
# Reward:
#   +1 for reaching the goal
#    0 for other transitions, including falling into a hole
# Termination:
#   Reaching the goal or falling into a hole.
# Time limit:
#   An episode can also end because of the step limit.
# is_slippery=False makes movement deterministic, so the
# agent's intended action determines its movement, subject
# to the grid boundaries and obstacles.
# Q-table dimensions: 64 states x 4 actions.

def make_env():
    return gym.make(
        ENV_NAME,
        map_name=MAP_NAME,
        is_slippery=False,
    )


# PART B: Q-learning implementation
def choose_action(q_table, state, epsilon, action_space):
    """Choose an action using epsilon-greedy exploration."""
    if np.random.random() < epsilon:
        return action_space.sample()

    return int(np.argmax(q_table[state]))


def train_q_learning(
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY,
    episodes=EPISODES,
    seed=42,
):
    """Train a tabular Q-learning agent."""

    env = make_env()

    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n

    q_table = np.zeros(
        (number_of_states, number_of_actions),
        dtype=float,
    )

    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    epsilon = epsilon_start

    for episode in range(episodes):
        state, info = env.reset(seed=seed + episode)

        total_reward = 0.0
        steps = 0
        success = False

        for step in range(MAX_STEPS):
            action = choose_action(
                q_table,
                state,
                epsilon,
                env.action_space,
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # Q-learning update:
            # Q(s,a) <- Q(s,a) + alpha *
            # [r + gamma * max_a' Q(s',a') - Q(s,a)]
            #
            # If the environment reaches a true terminal state,
            # there is no future value to bootstrap from.

            if terminated:
                target = reward
            else:
                target = (
                    reward
                    + gamma * np.max(q_table[next_state])
                )

            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            steps += 1

            if terminated or truncated:
                success = bool(reward == 1)
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_successes.append(int(success))

        # Reduce exploration as training progresses.
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay,
        )

        if (episode + 1) % 1000 == 0:
            recent_success = np.mean(
                episode_successes[-1000:]
            )
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Recent success rate: "
                f"{recent_success:.3f}"
            )

    env.close()

    history = {
        "rewards": np.array(episode_rewards),
        "lengths": np.array(episode_lengths),
        "successes": np.array(episode_successes),
    }

    return q_table, history


# Utility functions for plots and results
def moving_average(values, window=SMOOTHING_WINDOW):
    """Calculate a moving average for plotting."""
    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def plot_rewards(history, title, filename):
    """Plot reward versus training episode."""
    rewards = history["rewards"]
    smoothed = moving_average(rewards)

    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(1, len(rewards) + 1),
        rewards,
        alpha=0.25,
        label="Episode reward",
    )

    start_episode = len(rewards) - len(smoothed) + 1
    plt.plot(
        np.arange(start_episode, len(rewards) + 1),
        smoothed,
        label=f"Moving average ({SMOOTHING_WINDOW})",
    )

    plt.xlabel("Training episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_comparison(results, parameter_name, filename):
    """Plot moving-average rewards for multiple experiments."""
    plt.figure(figsize=(10, 5))

    for label, history in results.items():
        smoothed = moving_average(history["rewards"])
        start_episode = (
            len(history["rewards"]) - len(smoothed) + 1
        )

        plt.plot(
            np.arange(start_episode, len(history["rewards"]) + 1),
            smoothed,
            label=str(label),
        )

    plt.xlabel("Training episode")
    plt.ylabel("Moving-average reward")
    plt.title(f"Q-learning comparison: {parameter_name}")
    plt.legend(title=parameter_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def save_experiment_summary(results, filename):
    """Save average training statistics to a CSV file."""
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Setting",
            "Mean training reward",
            "Training success rate",
            "Mean episode length",
        ])

        for setting, history in results.items():
            writer.writerow([
                setting,
                np.mean(history["rewards"]),
                np.mean(history["successes"]),
                np.mean(history["lengths"]),
            ])


# PART C: Baseline training
def run_baseline():
    print("\nPART C: Baseline training")

    q_table, history = train_q_learning(
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        episodes=EPISODES,
        seed=42,
    )

    plot_rewards(
        history,
        "Baseline Q-learning: FrozenLake 8x8",
        os.path.join(RESULTS_DIR, "baseline_rewards.png"),
    )

    np.save(
        os.path.join(RESULTS_DIR, "baseline_q_table.npy"),
        q_table,
    )

    print("Baseline mean training reward:",
          np.mean(history["rewards"]))
    print("Baseline training success rate:",
          np.mean(history["successes"]))

    return q_table, history


# PART D: Learning-rate experiment
def run_learning_rate_experiment():
    print("\nPART D: Learning-rate experiment")

    learning_rates = [0.1, 0.5, 0.9]
    results = {}

    for alpha in learning_rates:
        print(f"\nTraining with alpha = {alpha}")

        _, history = train_q_learning(
            alpha=alpha,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY,
            episodes=EPISODES,
            seed=42,
        )

        results[f"alpha={alpha}"] = history

    plot_comparison(
        results,
        "Learning rate",
        os.path.join(RESULTS_DIR, "learning_rate_comparison.png"),
    )

    save_experiment_summary(
        results,
        os.path.join(RESULTS_DIR, "learning_rate_summary.csv"),
    )

    return results


# PART E: Discount-factor experiment
def run_discount_factor_experiment():
    print("\nPART E: Discount-factor experiment")

    discount_factors = [0.50, 0.90, 0.99]
    results = {}

    for gamma in discount_factors:
        print(f"\nTraining with gamma = {gamma}")

        _, history = train_q_learning(
            alpha=ALPHA,
            gamma=gamma,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY,
            episodes=EPISODES,
            seed=42,
        )

        results[f"gamma={gamma}"] = history

    plot_comparison(
        results,
        "Discount factor",
        os.path.join(RESULTS_DIR, "discount_factor_comparison.png"),
    )

    save_experiment_summary(
        results,
        os.path.join(RESULTS_DIR, "discount_factor_summary.csv"),
    )

    return results


# PART F: Exploration experiment
def run_exploration_experiment():
    print("\nPART F: Exploration-decay experiment")

    decay_rates = [0.99, 0.995, 0.999]
    results = {}

    for decay in decay_rates:
        print(f"\nTraining with epsilon decay = {decay}")

        _, history = train_q_learning(
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=decay,
            episodes=EPISODES,
            seed=42,
        )

        results[f"decay={decay}"] = history

    plot_comparison(
        results,
        "Epsilon decay",
        os.path.join(RESULTS_DIR, "exploration_comparison.png"),
    )

    save_experiment_summary(
        results,
        os.path.join(RESULTS_DIR, "exploration_summary.csv"),
    )

    return results


# PART G: Evaluate a trained policy
def evaluate_agent(q_table, episodes=EVALUATION_EPISODES):
    """Evaluate greedily, with exploration disabled."""

    env = make_env()

    rewards = []
    lengths = []
    successes = []

    for episode in range(episodes):
        state, info = env.reset(seed=100000 + episode)

        total_reward = 0.0
        steps = 0

        for step in range(MAX_STEPS):
            # Greedy action selection: epsilon = 0.
            action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            state = next_state
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(int(total_reward == 1))

    env.close()

    print("\nEvaluation results")
    print("------------------")
    print("Episodes:", episodes)
    print("Mean reward:", np.mean(rewards))
    print("Success rate:", np.mean(successes))
    print("Mean steps:", np.mean(lengths))

    return {
        "rewards": np.array(rewards),
        "lengths": np.array(lengths),
        "successes": np.array(successes),
    }


# Main program
def main():
    # Part C: Train the baseline agent.
    baseline_q, baseline_history = run_baseline()

    # Part D: Compare learning rates.
    learning_rate_results = run_learning_rate_experiment()

    # Part E: Compare discount factors.
    discount_results = run_discount_factor_experiment()

    # Part F: Compare exploration schedules.
    exploration_results = run_exploration_experiment()

    # Select the best configuration based on training success
    # rates. Evaluate each candidate greedily before deciding.
    candidates = {
        "baseline": baseline_q,
    }

    # Retain Q-tables from the experiments in a separate
    # selection run so the final candidate can be evaluated.
    configurations = [
        ("alpha=0.1", 0.1, GAMMA, EPSILON_DECAY),
        ("alpha=0.5", 0.5, GAMMA, EPSILON_DECAY),
        ("alpha=0.9", 0.9, GAMMA, EPSILON_DECAY),
        ("gamma=0.50", ALPHA, 0.50, EPSILON_DECAY),
        ("gamma=0.90", ALPHA, 0.90, EPSILON_DECAY),
        ("gamma=0.99", ALPHA, 0.99, EPSILON_DECAY),
        ("decay=0.99", ALPHA, GAMMA, 0.99),
        ("decay=0.995", ALPHA, GAMMA, 0.995),
        ("decay=0.999", ALPHA, GAMMA, 0.999),
    ]

    # Re-train candidates and compare their greedy evaluation.
    # This costs extra time but gives a consistent selection
    # method using the same evaluation episode count.
    for index, (name, alpha, gamma, decay) in enumerate(
        configurations
    ):
        print(f"\nSelecting candidate: {name}")

        q_table, _ = train_q_learning(
            alpha=alpha,
            gamma=gamma,
            epsilon_start=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=decay,
            episodes=EPISODES,
            seed=42,
        )

        candidates[name] = q_table

    evaluation_results = {}

    for name, q_table in candidates.items():
        print(f"\nEvaluating configuration: {name}")
        evaluation_results[name] = evaluate_agent(
            q_table,
            episodes=EVALUATION_EPISODES,
        )

    best_name = max(
        evaluation_results,
        key=lambda name: np.mean(
            evaluation_results[name]["rewards"]
        ),
    )

    best_q_table = candidates[best_name]

    print("\nBest configuration:", best_name)

    np.save(
        os.path.join(RESULTS_DIR, "best_q_table.npy"),
        best_q_table,
    )

    # Save evaluation results for the report.
    with open(
        os.path.join(RESULTS_DIR, "evaluation_summary.csv"),
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "Configuration",
            "Mean reward",
            "Success rate",
            "Mean steps",
        ])

        for name, result in evaluation_results.items():
            writer.writerow([
                name,
                np.mean(result["rewards"]),
                np.mean(result["successes"]),
                np.mean(result["lengths"]),
            ])

    print("\nAll plots and result summaries are in the results/ folder.")
    print("Remember to discuss the actual results in your report.")


if __name__ == "__main__":
    main()
