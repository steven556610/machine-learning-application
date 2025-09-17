import pandas as pd
import logging
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Assuming src.log_module.Logger and other modules exist in your project
from src.log_module import Logger
from src.path_setter import PathSetter

'''
import logging
from src.data_loader import DataLoader
from src.path_setter import PathSetter

if __name__ == '__main__':
    # 1. Load data
    data_loader = DataLoader(log_level=logging.INFO)
    df = data_loader.kaggle_df

    if df.empty:
        print("DataFrame is empty. Cannot proceed.")
    else:
        # 2. Instantiate and fit the Word2Vec transformer
        w2v_transformer = Word2VecTransformer(vector_size=128, window=5, min_count=2)
        w2v_transformer.fit(df, text_column='text_cleaned')

        # 3. Transform the DataFrame
        # This will add a new 'text_vector' column to the DataFrame
        df_transformed = w2v_transformer.transform(df, text_column='text_cleaned')
        
        # Display the first few rows to see the new vector column
        print(df_transformed[['text_cleaned', 'text_vector']].head())

        # 4. (Optional) Save the trained model
        model_save_path = os.path.join(PathSetter.script_dir, 'model', 'word2vec_model.bin')
        w2v_transformer.save_model(model_save_path)
'''

class Word2VecTransformer(Logger):
    """
    A class to train a Word2Vec model on a DataFrame and provide
    methods for text vectorization.
    """
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4, log_level=logging.INFO):
        """
        Initializes the Word2VecTransformer with training parameters.

        Args:
            vector_size (int): The dimensionality of the word vectors.
            window (int): The maximum distance between the current and predicted word.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker threads to train the model.
            log_level (int): The logging level for this class.
        """
        super().__init__(log_level=log_level)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, df: pd.DataFrame, text_column: str = 'text'):
        """
        Trains the Word2Vec model on the specified text column of a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the column containing the text data.

        Returns:
            Word2VecTransformer: The fitted transformer instance.
        """
        if text_column not in df.columns:
            self.logger.error(f"Text column '{text_column}' not found in the DataFrame.")
            raise ValueError(f"DataFrame must contain a column named '{text_column}'.")

        self.logger.info("Tokenizing text for Word2Vec training.")
        # Preprocess each document into a list of words.
        # simple_preprocess handles tokenization and lowercasing.
        documents = df[text_column].apply(lambda x: simple_preprocess(x) if isinstance(x, str) else [])

        self.logger.info("Starting Word2Vec model training.")
        self.model = Word2Vec(
            sentences=documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        self.logger.info("Word2Vec model training complete.")
        return self

    def get_document_vector(self, text_list: list) -> list:
        """
        Calculates the average word vector for a list of words (a document).

        Args:
            text_list (list): A list of words representing a document.

        Returns:
            list: The average vector for the document, or a zero vector if no words are in the model's vocabulary.
        """
        if self.model is None:
            self.logger.warning("Model not trained. Returning zero vector.")
            return [0] * self.vector_size

        vectors = []
        for word in text_list:
            try:
                vectors.append(self.model.wv[word])
            except KeyError:
                continue # Skip words not in the vocabulary

        if not vectors:
            return [0] * self.vector_size
        
        # Calculate the average vector
        avg_vector = sum(vectors) / len(vectors)
        return avg_vector

    def transform(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Transforms the text column of a DataFrame into a new column of Word2Vec vectors.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_column (str): The name of the column to transform.

        Returns:
            pd.DataFrame: A new DataFrame with the added 'text_vector' column.
        """
        if self.model is None:
            self.logger.error("Model has not been trained. Call .fit() first.")
            raise RuntimeError("Model must be fitted before transforming data.")
        
        self.logger.info("Transforming text to vectors.")
        
        # Preprocess the text in the same way as during training
        documents = df[text_column].apply(lambda x: simple_preprocess(x) if isinstance(x, str) else [])
        
        # Apply the vectorization method to each document
        df['text_vector'] = documents.apply(self.get_document_vector)
        self.logger.info("Transformation complete.")
        return df
    
    def save_model(self, path: str):
        """
        Saves the trained Word2Vec model to a file.
        """
        if self.model:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
        else:
            self.logger.warning("No model to save. Please train the model first.")

    def load_model(self, path: str):
        """
        Loads a Word2Vec model from a file.
        """
        try:
            self.model = Word2Vec.load(path)
            self.logger.info(f"Model loaded from {path}")
            return self
        except FileNotFoundError:
            self.logger.error(f"Model file not found at: {path}")
            return None