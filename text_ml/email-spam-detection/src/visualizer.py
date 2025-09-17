# visualizer.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import logging

from src.log_module import Logger

'''
# Example Usage
if __name__ == '__main__':
    from src.data_loader import DataLoader
    
    # Assuming you have a DataLoader class that returns a DataFrame
    data_loader = DataLoader()
    balanced_data = data_loader.kaggle_df
    
    # 1. Initialize the Visualizer with the DataFrame
    viz = Visualizer(df=balanced_data, log_level=logging.INFO)
    
    # 2. Call the plotting methods
    # We pass 'ham' and 'spam' as the labels to filter the data
    viz.plot_word_cloud(label='ham')
    viz.plot_word_cloud(label='spam')
'''

class Visualizer(Logger):
    """
    A class to handle data visualization for the project.
    It encapsulates plotting functions to keep the main script clean.
    """
    def __init__(self, df: pd.DataFrame, log_level=logging.INFO):
        """
        Initializes the Visualizer with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data to visualize.
            log_level (int): The logging level for this class.
        """
        super().__init__(log_level=log_level)
        self.df = df
        self.logger.info("Visualizer initialized.")

    def plot_word_cloud(self, label: str, text_col: str = 'text', label_col: str = 'label'):
        """
        Generates and displays a word cloud for a specific category of emails.

        Args:
            label (str): The label (e.g., 'ham' or 'spam') to filter the data.
            text_col (str): The name of the column containing the text data.
            label_col (str): The name of the column containing the labels.
        """
        self.logger.info(f"Generating word cloud for '{label}' emails.")
        
        # Filter the DataFrame for the specified label
        data_subset = self.df[self.df[label_col] == label]
        
        if data_subset.empty:
            self.logger.warning(f"No data found for label '{label}'. Skipping word cloud generation.")
            return

        # Combine all text into a single string
        email_corpus = " ".join(data_subset[text_col])

        # Generate the word cloud
        wordcloud = WordCloud(
            background_color='black', 
            max_words=100, 
            width=800, 
            height=400
        ).generate(email_corpus)

        # Plotting
        plt.figure(figsize=(7, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'WordCloud for {label.capitalize()} Emails', fontsize=15)
        plt.axis('off')
        plt.show()

