import pandas as pd
import numpy as np
import pickle
import os
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.log_module import Logger
from src.path_setter import PathSetter

'''
import logging
from sklearn.model_selection import train_test_split
from src.data_loader import DataLoader
from src.onehot_encoder import TextTransformer

if __name__ == "__main__":
    # Load data
    data_loader = DataLoader(log_level=logging.INFO)
    df = data_loader.kaggle_df

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_X = train_df['text_cleaned']
    test_X = test_df['text_cleaned']

    # 1. Fit and transform the data
    transformer = TextTransformer(max_len=50).fit(train_X)
    train_sequences = transformer.transform(train_X)
    test_sequences = transformer.transform(test_X)

    print("\nShape of training sequences:", train_sequences.shape)
    print("Shape of testing sequences:", test_sequences.shape)

    # 2. (Optional) Save the fitted transformer
    save_path = os.path.join(PathSetter.script_dir, 'model', 'tokenizer.pkl')
    transformer.save_transformer(save_path)

    # 3. (Optional) Load the saved transformer to reuse it later
    loaded_transformer = TextTransformer().load_transformer(save_path)
    if loaded_transformer:
        re_transformed_sequences = loaded_transformer.transform(test_X)
        print("\nShape of re-transformed sequences (from loaded file):", re_transformed_sequences.shape)
        # Verify that the loaded transformer produces the same results
        print("Are the sequences identical?", np.array_equal(test_sequences, re_transformed_sequences))
'''


class TextTransformer(Logger):
    """
    A class to handle text tokenization, padding, and one-hot encoding for deep learning.
    This transformer can be fitted, used to transform data, and saved/loaded.
    """
    def __init__(self, max_len=100, log_level=logging.INFO):
        """
        Initializes the TextTransformer.

        Args:
            max_len (int): The maximum length for sequences.
            log_level (int): The logging level for this class.
        """
        super().__init__(log_level=log_level)
        self.tokenizer = None
        self.max_len = max_len
        self.vocab_size = 0

    def fit(self, texts: pd.Series):
        """
        Fits the tokenizer on the provided text data.

        Args:
            texts (pd.Series): A pandas Series containing the text data.
        """
        self.logger.info("Fitting tokenizer on the provided text data.")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(texts)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.logger.info(f"Tokenizer fitted. Vocabulary size: {self.vocab_size}")
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transforms text into padded sequences.

        Args:
            texts (pd.Series): A pandas Series containing the text data to transform.

        Returns:
            np.ndarray: A NumPy array of the padded sequences.
        """
        if self.tokenizer is None:
            self.logger.error("Tokenizer not fitted. Please call .fit() first.")
            raise RuntimeError("Tokenizer must be fitted before transforming data.")

        self.logger.info("Converting texts to sequences and applying padding.")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding='post',
            truncating='post'
        )
        self.logger.info(f"Texts transformed. Output shape: {padded_sequences.shape}")
        return padded_sequences

    def save_transformer(self, file_path: str):
        """
        Saves the fitted tokenizer to a pickle file.

        Args:
            file_path (str): The path to save the tokenizer pickle file.
        """
        if self.tokenizer is None:
            self.logger.warning("No tokenizer to save. Please fit the transformer first.")
            return

        with open(file_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        self.logger.info(f"Tokenizer saved to {file_path}")

    def load_transformer(self, file_path: str):
        """
        Loads a pre-fitted tokenizer from a pickle file.

        Args:
            file_path (str): The path to the tokenizer pickle file.
        
        Returns:
            TextTransformer: The loaded TextTransformer instance.
        """
        try:
            with open(file_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.vocab_size = len(self.tokenizer.word_index) + 1
            self.logger.info(f"Tokenizer loaded from {file_path}. Vocabulary size: {self.vocab_size}")
            return self
        except FileNotFoundError:
            self.logger.error(f"Tokenizer file not found at: {file_path}")
            return None

    def get_tokenizer(self):
        """Returns the fitted Keras Tokenizer object."""
        return self.tokenizer
    
    # --- For true one-hot encoding (not recommended for most DL tasks) ---
    def transform_to_onehot(self, texts: pd.Series) -> np.ndarray:
        """
        Transforms text into a one-hot encoded matrix.
        
        Warning: This is not ideal for deep learning models that use embedding layers.
        It's better to use `transform()` and an embedding layer.
        """
        if self.tokenizer is None:
            self.logger.error("Tokenizer not fitted. Cannot perform one-hot encoding.")
            raise RuntimeError("Tokenizer must be fitted before one-hot encoding.")
            
        self.logger.info("Converting texts to one-hot encoded matrix.")
        one_hot_matrix = self.tokenizer.texts_to_matrix(texts, mode='binary')
        self.logger.info(f"Texts one-hot encoded. Output shape: {one_hot_matrix.shape}")
        return one_hot_matrix