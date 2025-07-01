import pandas as pd
from pathlib import Path
from typing import List

class TextDataLoader:
    """Class for loading text datasets for authorship recognition."""
    
    def __init__(self, data_dir: str):
        """
        Initializes TextDataLoader with the directory containing JSON files.
        
        Args:
            data_dir (str): Directory containing train_data.json, val_data.json, and test_data.json
        """
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_train(self) -> None:
        """Loads the training dataset from train_data.json."""
        try:
            self.train_data = pd.read_json(self.data_dir / 'train_data.json')
            print(f"Loaded train dataset: {len(self.train_data)} works, {self.train_data['author'].nunique()} authors")
            print(f"\nAuthor distribution in Train set:")
            print(self.train_data['author'].value_counts())
        except Exception as e:
            print(f"Error loading train_data.json: {e}")
            self.train_data = pd.DataFrame()

    def load_val(self) -> None:
        """Loads the validation dataset from val_data.json."""
        try:
            self.val_data = pd.read_json(self.data_dir / 'val_data.json')
            print(f"Loaded validation dataset: {len(self.val_data)} works, {self.val_data['author'].nunique()} authors")
            print(f"\nAuthor distribution in Validation set:")
            print(self.val_data['author'].value_counts())
        except Exception as e:
            print(f"Error loading val_data.json: {e}")
            self.val_data = pd.DataFrame()

    def load_test(self) -> None:
        """Loads the test dataset from test_data.json."""
        try:
            self.test_data = pd.read_json(self.data_dir / 'test_data.json')
            print(f"Loaded test dataset: {len(self.test_data)} works, {self.test_data['author'].nunique()} authors")
            print(f"\nAuthor distribution in Test set:")
            print(self.test_data['author'].value_counts())
        except Exception as e:
            print(f"Error loading test_data.json: {e}")
            self.test_data = pd.DataFrame()

    def get_raw_data(self, split: str = 'train') -> pd.DataFrame:
        """
        Returns the raw DataFrame for a specified split.
        
        Args:
            split (str): Dataset split to return ('train', 'val', or 'test')
        
        Returns:
            pd.DataFrame: Raw data for the specified split
        """
        if split == 'train':
            if self.train_data is None:
                print("Train dataset not loaded. Call load_train() first.")
                return pd.DataFrame()
            return self.train_data
        elif split == 'val':
            if self.val_data is None:
                print("Validation dataset not loaded. Call load_val() first.")
                return pd.DataFrame()
            return self.val_data
        elif split == 'test':
            if self.test_data is None:
                print("Test dataset not loaded. Call load_test() first.")
                return pd.DataFrame()
            else:
                raise ValueError("Split must be 'train', 'val', or 'test'")

    def get_author_list(self) -> List[str]:
        """Returns the list of unique authors from loaded datasets."""
        authors = set()
        for data in [self.train_data, self.val_data, self.test_data]:
            if data is not None and not data.empty:
                authors.update(data['author'].unique())
        return sorted(list(authors))

def main():
    """Example usage of TextDataLoader."""
    DATA_DIR = '../data'
    loader = TextDataLoader(DATA_DIR)
    
    print("Loading training data...")
    loader.load_train()
    
    print("\nLoading validation data...")
    loader.load_val()
    
    print("\nLoading test data...")
    loader.load_test()
    
    print("\nAuthors:", loader.get_author_list())
 
    train_data = loader.get_raw_data('train')
    print(train_data.head())
    print(train_data.head()['text'])

if __name__ == "__main__":
    main()