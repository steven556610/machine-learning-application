import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging

from src.log_module import Logger
from src.model_definer import SmallTransformerClassifier
from src.word2vec_transformer import Word2VecTransformer

class ModelBuilder(Logger):
    def __init__(self, hyperparams: dict, log_level=logging.INFO):
        """
        Initializes the ModelBuilder with hyperparameters.

        Args:
            hyperparams (dict): A dictionary of training hyperparameters.
            log_level (int): The logging level for this class.
        """
        super().__init__(log_level=log_level)
        self.hyperparams = hyperparams
        self.model = None

    def build_model(self, vocab_size: int, d_model: int, output_dim: int):
        """
        Builds and returns a SmallTransformerClassifier model.

        Args:
            vocab_size (int): The size of the vocabulary from the tokenizer.
            d_model (int): The dimension of the word vectors.
            output_dim (int): The number of output classes.

        Returns:
            SmallTransformerClassifier: The configured model.
        """
        self.logger.info("Building the Transformer model.")
        self.model = SmallTransformerClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=self.hyperparams.get('nhead', 4),
            num_encoder_layers=self.hyperparams.get('num_encoder_layers', 2),
            dim_feedforward=self.hyperparams.get('dim_feedforward', 128),
            dropout=self.hyperparams.get('dropout', 0.1),
            output_dim=output_dim,
        )
        return self.model

    def get_callbacks(self):
        """
        Defines and returns a list of PyTorch Lightning callbacks.
        """
        self.logger.info("Setting up Early Stopping callback.")
        early_stopping = EarlyStopping(
            monitor='val_loss', # The metric to monitor
            mode='min',         # Stop when the metric stops decreasing
            patience=self.hyperparams.get('patience', 5), # Number of epochs to wait
            verbose=True,       # Print messages when stopping
        )
        return [early_stopping]

    def build_trainer(self):
        """
        Builds and returns a PyTorch Lightning Trainer.
        """
        self.logger.info("Building the PyTorch Lightning Trainer.")
        trainer = pl.Trainer(
            max_epochs=self.hyperparams.get('epochs', 10),
            callbacks=self.get_callbacks(),
            accelerator='auto', # Use GPU if available
        )
        return trainer

    def train_model(self, train_loader, val_loader):
        """
        Trains the model using the built trainer.
        """
        if self.model is None:
            self.logger.error("Model not built. Call .build_model() first.")
            return

        trainer = self.build_trainer()
        trainer.fit(self.model, train_loader, val_loader)