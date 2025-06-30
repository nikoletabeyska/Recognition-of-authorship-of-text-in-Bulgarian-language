import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split

class TextDataProcessor:
    """Class for processing and balancing text data for Bulgarian authors."""
    
    def __init__(self, data_json_path: str, text_dir: str, output_dir: str):
        """
        Initializes TextDataProcessor with file paths.
        
        Args:
            data_json_path (str): Path to the JSON file with metadata
            text_dir (str): Directory containing text files
            output_dir (str): Directory to save processed JSON files
        """
        self.data_json_path = data_json_path
        self.text_dir = text_dir
        self.output_dir = Path(output_dir)
        self.bulgarian_authors = [
            'Елин Пелин', 'Стефан Бонев', 'Христо Пощаков', 'Йордан Радичков',
            'Дончо Цончев', 'Ангел Каралийчев', 'Чудомир', 'Йордан Йовков',
            'Красимир Бачков', 'Алеко Константинов', 'Иван Вазов',
            'Николай Райнов', 'Любен Дилов', 'Мартин Дамянов', 'Агоп Мелконян'
        ]
        self.allowed_forms = [
            'Разказ', 'Очерк', 'Писмо', 'Есе', 'Фейлетон', 'Литературна критика', 
            'Рецензия', 'Роман', 'Новела', 'Повест', 'Приказка', 'Биография', 
            'Мемоари/спомени', 'Пътепис'
        ]
        self.data = None
        self.prose_data = None
        self.balanced_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_metadata(self) -> None:
        print("Loading data.json...")
        try:
            self.data = pd.read_json(self.data_json_path).transpose()
            self.data['id'] = self.data['meta'].apply(lambda x: x['id'])
            print("\nColumns in DataFrame:", self.data.columns.tolist())
        except Exception as e:
            print(f"Error loading data.json: {e}")
            raise

    def filter_prose(self) -> None:
        self.prose_data = self.data[
            (self.data['form'].isin(self.allowed_forms)) & 
            (self.data['author'].isin(self.bulgarian_authors))
        ]

    def process_text(self, text_path: str, min_words: int = 1000, target_words: int = 1000) -> Tuple[int, str]:
        """
        Processes a text file and returns word count and truncated text.
        
        Args:
            text_path (str): Path to the text file
            min_words (int): Minimum number of words
            target_words (int): Target number of words for truncation
            
        Returns:
            Tuple[int, str]: Word count and processed text
        """
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
                words = text.split()
                word_count = len(words)
                if word_count >= min_words:
                    return word_count, " ".join(words[:target_words])
                return word_count, ""
        except Exception as e:
            print(f"Error reading {text_path}: {e}")
            return 0, ""

    def process_texts(self) -> None:
        """Processes all texts and adds word count and text information."""
        print("\nProcessing texts...")
        word_counts = []
        texts = []
        for work_id in self.prose_data['id']:
            text_path = Path(self.text_dir) / f"{work_id}" / "text.txt"
            word_count, text = self.process_text(text_path)
            word_counts.append(word_count)
            texts.append(text)

        self.prose_data = self.prose_data.copy()
        self.prose_data['word_count'] = word_counts
        self.prose_data['text'] = texts

        # Filter for works with >=1000 words
        self.prose_data = self.prose_data[self.prose_data['word_count'] >= 1000]

    def create_balanced_dataset(self) -> None:
        """Creates a balanced dataset with 20 works per author."""
        author_counts = self.prose_data['author'].value_counts()
        eligible_authors = author_counts[author_counts >= 20].index

        self.balanced_data = pd.DataFrame()
        for author in eligible_authors:
            author_works = self.prose_data[self.prose_data['author'] == author]
            sampled_works = author_works.sample(n=20, random_state=42)
            self.balanced_data = pd.concat([self.balanced_data, sampled_works], ignore_index=True)

        # Ensure exactly 1000 words per text
        self.balanced_data['text'] = self.balanced_data['text'].apply(lambda x: " ".join(x.split()[:1000]))
        self.balanced_data['word_count'] = self.balanced_data['text'].apply(lambda x: len(x.split()))

        print(f"\nFinal balanced dataset: {len(self.balanced_data)} works, "
              f"{self.balanced_data['author'].nunique()} authors")
        print(f"Word count statistics:\n{self.balanced_data['word_count'].describe()}")

    def split_dataset(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15) -> None:
        """
        Splits the balanced dataset into stratified train, validation, and test sets.
        
        Args:
            train_size (float): Proportion of data for training (default: 0.7)
            val_size (float): Proportion of data for validation (default: 0.15)
            test_size (float): Proportion of data for testing (default: 0.15)
        """
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        print("\nSplitting dataset into train, validation, and test sets...")
        train_val, self.test_data = train_test_split(
            self.balanced_data,
            test_size=test_size,
            stratify=self.balanced_data['author'],
            random_state=42
        )

        relative_val_size = val_size / (train_size + val_size)
        self.train_data, self.val_data = train_test_split(
            train_val,
            test_size=relative_val_size,
            stratify=train_val['author'],
            random_state=42
        )

        print(f"Training set: {len(self.train_data)} works, {self.train_data['author'].nunique()} authors")
        print(f"Validation set: {len(self.val_data)} works, {self.val_data['author'].nunique()} authors")
        print(f"Test set: {len(self.test_data)} works, {self.test_data['author'].nunique()} authors")

        for split_name, split_data in [("Train", self.train_data), ("Validation", self.val_data), ("Test", self.test_data)]:
            print(f"\nAuthor distribution in {split_name} set:")
            print(split_data['author'].value_counts())

    def save_balanced_dataset(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in [("train", self.train_data), ("val", self.val_data), ("test", self.test_data)]:
            if split_data is not None:
                json_data = [
                    {"id": row['id'], "author": row['author'], "text": row['text']}
                    for _, row in split_data.iterrows()
                ]
                output_path = self.output_dir / f"{split_name}_data.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"\n{split_name.capitalize()} dataset saved to: {output_path}")

    def run(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15) -> None:
        """
        Executes the full processing pipeline, including stratified splitting.
        
        Args:
            train_size (float): Proportion of data for training (default: 0.7)
            val_size (float): Proportion of data for validation (default: 0.15)
            test_size (float): Proportion of data for testing (default: 0.15)
        """
        self.load_metadata()
        self.filter_prose()
        self.process_texts()
        self.create_balanced_dataset()
        self.split_dataset(train_size, val_size, test_size)
        self.save_balanced_dataset()

def main():
    DATA_JSON_PATH = '../data/raw/data.json'
    TEXT_DIR = '../data/raw'
    OUTPUT_DIR = '../data/processed'

    processor = TextDataProcessor(DATA_JSON_PATH, TEXT_DIR, OUTPUT_DIR)
    processor.run(train_size=0.7, val_size=0.15, test_size=0.15)

if __name__ == "__main__":
    main()