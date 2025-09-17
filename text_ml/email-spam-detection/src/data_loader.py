import os
import re
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords

from src.log_module import Logger
from src.path_setter import PathSetter

# Download NLTK stopwords only once
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
    

'''
# Example Usage
if __name__ == '__main__':
    # Set the log level for this module
    data_loader = DataLoader(log_level=logging.DEBUG)

    if not data_loader.kaggle_df.empty:
        print("Kaggle DataFrame Head:")
        print(data_loader.kaggle_df.head())
        print("-" * 30)

    if not data_loader.geek_df.empty:
        print("GeeksforGeeks DataFrame Head:")
        print(data_loader.geek_df.head())
'''

class DataLoader(Logger):
    def __init__(self):
        super().__init__()
        self.stop_words = set(stopwords.words('english'))
        
        self.logger.info("Loading and processing Kaggle dataset.")
        self.kaggle_df = self._process_kaggle_data()
        
        self.logger.info("Loading and processing GeeksforGeeks dataset.")
        self.geek_df = self._process_geek_data()
        
    def _process_kaggle_data(self):
        """
        Loads, cleans, and processes the Kaggle dataset.
        """
        try:
            df = pd.read_csv(PathSetter.kaggle_data_path)
            # Standardize column names
            df.rename(columns={'Category': 'label', 'Masseges': 'text'}, inplace=True)
            
            # Map 'ham' to 0 and 'spam' to 1 for numerical representation
            df['label_num'] = df['label'].apply(lambda x: 0 if x == 'ham' else 1)
            
            # Remove "Subject:" from the beginning of text
            df['text'] = df['text'].str.replace('Subject', '', regex=False).str.strip()
            
            # Apply cleaning and stopword removal
            df['text_cleaned'] = df['text'].apply(self._clean_text)
            df['text_cleaned'] = df['text_cleaned'].apply(self._remove_stopwords)
            self.logger.info("Kaggle data processing complete.")
            return df
        except FileNotFoundError:
            self.logger.error(f"Kaggle data file not found at: {PathSetter.kaggle_data_path}")
            return pd.DataFrame()

    def _process_geek_data(self):
        """
        Loads, cleans, and processes the GeeksforGeeks dataset.
        """
        try:
            df = pd.read_csv(PathSetter.geek_data_path)
            # Clean text and remove stopwords for the entire DataFrame column
            df['text'] = df['text'].apply(self._clean_text)
            df['text_cleaned'] = df['text'].apply(self._remove_stopwords)
            self.logger.info("GeeksforGeeks data processing complete.")
            return df
        except FileNotFoundError:
            self.logger.error(f"GeeksforGeeks data file not found at: {PathSetter.geek_data_path}")
            return pd.DataFrame()

    def _clean_text(self, text):
        """
        Cleans text by removing punctuation and converting to lowercase.
        """
        if not isinstance(text, str):
            text = str(text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        return text.lower().strip()

    def _remove_stopwords(self, text):
        """
        Removes stopwords from a given text string.
        """
        # A more efficient approach using list comprehension and join
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return " ".join(filtered_words)

