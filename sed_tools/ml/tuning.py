"""
Hyperparameter search for the SED generator (``sed-tools ml_generator --auto``).

Each optuna trial trains a SEDGenerator network briefly on the library; the
network shape, dropout and learning rate with the lowest validation loss are
then trained in full by SEDGenerator.train, so the saved model is an ordinary
SEDGenerator model.

Needs optuna: ``pip install 'sed-tools[ml-tune]'``.
"""

import optuna

from .generator import SEDGenerator, SEDGeneratorNetwork, _optimizer_and_scheduler
from .common import data_loaders, fit

# Epochs each trial trains for before its validation loss is compared
TRIAL_EPOCHS = 10
# SEDGenerator.train's defaults; trials and the final training both use them
BATCH_SIZE = 64
MAX_SAMPLES = 10000


def _hidden_layers(params):
    """The SEDGenerator hidden layers for a trial's parameters: num_blocks layers of hidden_dim."""
    return [params["hidden_dim"]] * params["num_blocks"]


class AutoTrainer:
    """Tune SEDGenerator's hyperparameters with optuna, then train it with the best."""
    def __init__(self, library_path: str, task: str = "generator"):
        from .common import get_device

        self.library_path = library_path
        self.task = task
        self.device = get_device()
        print(f"Using device: {self.device}")

    def objective(self, trial, train_loader, val_loader, output_dim):
        """Train one trial's network for TRIAL_EPOCHS and return its best validation loss."""
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 1024, step=128),
            "num_blocks": trial.suggest_int("num_blocks", 1, 5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        }
        model = SEDGeneratorNetwork(output_dim, hidden_layers=_hidden_layers(params), dropout=params["dropout"])
        optimizer, scheduler = _optimizer_and_scheduler(model, params["lr"])
        _, best_val_loss = fit(
            model, train_loader, val_loader, optimizer, scheduler,
            epochs=TRIAL_EPOCHS, patience=TRIAL_EPOCHS, verbose=False, device=self.device, progress="",
        )
        return best_val_loss

    def train_best(self, library_path, output_path, n_trials=20, epochs=100):
        """
        Run ``n_trials`` trials, then train a SEDGenerator with the best
        parameters for up to ``epochs`` epochs and save it to ``output_path``.

        Returns the trained SEDGenerator and the best parameters
        (hidden_dim, num_blocks, lr, dropout).
        """
        train_set, val_set = SEDGenerator()._training_split(library_path, MAX_SAMPLES)
        train_loader, val_loader = data_loaders(train_set, val_set, BATCH_SIZE)
        output_dim = val_set[1].shape[1]

        print(f"Starting hyperparameter optimization ({n_trials} trials)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial, train_loader, val_loader, output_dim), n_trials=n_trials)
        best_params = study.best_params
        print("Best hyperparameters:", best_params)

        print(f"Training final model for up to {epochs} epochs...")
        generator = SEDGenerator()
        generator.train(
            library_path, output_path,
            epochs=epochs, batch_size=BATCH_SIZE, learning_rate=best_params["lr"],
            max_samples=MAX_SAMPLES, hidden_layers=_hidden_layers(best_params),
            dropout=best_params["dropout"],
        )
        return generator, best_params
