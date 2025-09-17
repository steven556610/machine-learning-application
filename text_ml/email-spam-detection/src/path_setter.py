import os

class PathSetter:
    """
    A class to store and manage project-related file paths.

    Using this class helps to centralize path definitions and makes them
    easily accessible and modifiable in one place.
    """
    
    # Get the directory where the script is located
    # This makes the paths relative and portable across different machines.
    # os.path.dirname gets the directory name from a path.
    # os.path.abspath gets the absolute path, which is safer.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use os.path.join to create platform-independent paths.
    # It automatically handles the correct slashes ('/' or '\').
    
    # Raw data paths
    geek_data_path = os.path.join(script_dir, 'data', 'spam_ham_dataset_geeksforgeeks.csv')
    kaggle_data_path = os.path.join(script_dir, 'data', 'spam mail.csv')
    
    # Tokenizer path
    tokenizer_path = os.path.join(script_dir, 'model', 'geeksforgeeks_tokenizer.pkl')
    
    # Model path
    model_path = os.path.join(script_dir, 'model', 'geeksforgeeks_emailspam.h5')

    def __init__(self):
        # A simple __init__ method to show the class can be instantiated.
        # You can add checks here to ensure all paths exist.
        pass