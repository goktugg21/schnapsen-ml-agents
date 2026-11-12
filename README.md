# Comparing Neural Networks and Logistic Regression for Schnapsen Agents

Course project for Project Intelligent Systems (2025). The project compares two machine learning agents for the card game Schnapsen: a neural network (a multilayer perceptron) and logistic regression. Both agents are trained on game data produced by bots playing against each other, then tested against five opponents. The report finds that the neural network improves with more training data and beats the hand-written Phase2FocusedBot at 20,000 training games, while logistic regression is stronger than the neural network against the simple Bully bot. Both models do poorly against the RDeep lookahead bots.

Full write-up: [report/Comparing_NN_and_LR_for_Schnapsen.pdf](report/Comparing_NN_and_LR_for_Schnapsen.pdf)

## Method

- Training data: self-play games between three bots (Phase2FocusedBot, RDeep and Bully), recorded with `MLDataBot`. Training sets of 2,000 to 20,000 games were used (2,000, 5,000, 8,000, 10,000, 15,000 and 20,000).
- Models: a multilayer perceptron with one hidden layer of 30 tanh units (with early stopping), and logistic regression. Both are trained with the course platform's `train_ML_model`.
- Evaluation: 200 position-balanced games per matchup (100 as first player, 100 as second player) against five opponents: Random, Bully, Phase2FocusedBot, RDeep-Basic and RDeep-Advanced.
- Phase2FocusedBot is a hand-written bot in this repo. It prefers marriages and trump exchanges in phase 1, and plays for points in phase 2.

## Results (numbers as stated in the report)

Win rates are over 200 games per matchup.

| Measure | Neural network | Logistic regression |
|---|---|---|
| Win rate vs Phase2FocusedBot | 36.0% (2,000 training games), 64.5% (20,000) | 45% to 53% across training sizes |
| Average win rate, best | 58.9% (20,000 games) | 54.2% (10,000 games) |
| vs Random | 78.5% (20,000) | 65% on average, 66.5% at 10,000 |
| vs Bully | 79.5% (20,000) | up to 88% (20,000) |
| vs RDeep-Basic | 37.0% | up to 42.5% |
| vs RDeep-Advanced | 35.0% | 26.5% |
| Training time | 0.081 min (2,000 games), 0.907 min (10,000), 1.478 min (20,000); about 0.73 min on average | about 0.34 min on average; 0.076 min at 2,000 games |

The report does not give a logistic regression training time for 20,000 games, only the averages above.

## How to run

The code is not a standalone package. It runs on top of the course platform, which is not included here: <https://github.com/intelligent-systems-course/schnapsen>.

1. Clone the platform repository and install its requirements (see its README).
2. Copy `schnapsen_ml_models.py` from this repo into the root of the cloned platform repository, so that `schnapsen` is importable.
3. From that folder, run:

   ```bash
   python schnapsen_ml_models.py
   ```

The script creates `ML_replay_memories/` and `ML_models/` in the working directory (and deletes them first if they exist), plays the training games, trains the neural network and logistic regression models, and plays 200 test games per opponent. It also plays the two models against each other. By default it uses 10,000 training games. To reproduce the other training sizes, change `num_games` in the `create_training_data(...)` call at the bottom of the file.

Generated replay memories and models are not committed (see `.gitignore`).

## Contents

- `schnapsen_ml_models.py`: `Phase2FocusedBot`, `create_training_data`, `train_models`, `play_test_games`, `test_models`.
- `report/Comparing_NN_and_LR_for_Schnapsen.pdf`: the project report.

## Author

Göktuğ Yıldırım — Project Intelligent Systems, BSc Artificial Intelligence, Vrije Universiteit Amsterdam (2025)

## License

MIT, for the code in this repository (see `LICENSE`). The Schnapsen platform is separate and has its own license.
