from dataset_loader import TextDataLoader
import fasttext
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict
import wandb


class TextDataset(Dataset):    
    def __init__(self, texts: List[str], labels: List[int], word_to_index: Dict[str, int], max_length: int = 1000):
        """
        Initialize TextDataset class for tokenization and sequence padding of text data.

        Parameters:
            texts (List[str]): List of texts to be tokenized.
            labels (List[int]): List of labels corresponding to each text.
            word_to_index (Dict[str, int]): Dictionary mapping words to their index in the vocabulary.
            max_length (int): Maximum length of text sequences. It is used for padding and/or truncating.
        """
        self.texts = texts
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_length = max_length
        
    def __len__(self) -> int:
        """
        Get number of texts in the dataset
        """
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert text to indices. Pad or truncate text where necessary.

        Parameters:
            index (int): Index of the text sample to retrieve.
        """
        text = self.texts[index]
        label = self.labels[index]

        tokens = text.split()[:self.max_length]
        indices = [self.word_to_index.get(token, self.word_to_index['<UNK>']) for token in tokens]

        if len(indices) < self.max_length:
            indices.extend([self.word_to_index['<PAD>']] * (self.max_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, 
                    num_classes: int, dropout: float, embedding_matrix: torch.Tensor):
        """
        Initialize LSTM model for text classification. Attention mechanism is included.

        Parameters:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Number of dimensions of word embeddings.
            hidden_dim (int): Number of dimensions of LSTM hidden state.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate for regularization.
            embedding_matrix (torch.Tensor): Pre-trained embedding matrix.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, 
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the LSTM model.

        Parameters:
            x (torch.Tensor): Input tensor of token indices.
        """
        mask = (x != 0).float()
        embedded = self.embedding(x)

        lstm_out, _ = self.lstm(embedded)

        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attention_weights = attention_weights * mask.unsqueeze(-1)

        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Classification
        context = self.batch_norm(context)
        context = self.dropout(context)
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        output = self.fc2(out)

        return output


class LSTMClassifier:
    def __init__(self, data_loader: TextDataLoader, device: str = 'auto'):
        """
        Initialize LSTMClassifier class that applies the LSTMModel class for the authorship classification problem.

        Parameters:
            data_loader (TextDataLoader): Instance of TextDataLoader.
            device (str): 'auto', 'cpu', 'cuda'.
        """
        self.data_loader = data_loader

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.label_encoder = LabelEncoder()
        self.word_to_index = None
        self.embedding_matrix = None
        self.authors = None
        
        # Most of the hyperparameters are taken from the best run with Optuna and Cross-Validation
        self.best_params = {
            'batch_size': 16,
            'dropout': 0.5, # Changed from 0.31848438034467685
            'epochs': 30,
            'hidden_dim': 64,
            'learning_rate': 0.002178593832584739,
            'max_length': 1000,
            'num_layers': 1,
            'weight_decay': 0.001 # Changed from 0.0000989258738679434
        }

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text.

        Parameters:
            text (str): Raw text.
        """
        # Convert text to lower case
        text = text.lower()

        # Replace all characters except Cyrillic and Latin letters, whitespace, and specific punctuation with a single space.
        text = re.sub(r'[^\u0410-\u044f\u0401\u0451a-zA-Z\s.,!?;:"-]', ' ', text)

        # Replace multiple consecutive spaces with a single space and remove leading and trailing spaces.
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def load_pretrained_embeddings(self, texts: List[str]) -> None:
        """
        Load pre-trained Bulgarian FastText embeddings for vocabulary.
        
        Parameters:
            texts (List[str]): List of texts to build vocabulary from.
        """
        print("Loading pre-trained Bulgarian FastText embeddings...")

        model_path = 'cc.bg.300.bin/cc.bg.300.bin'
        words_vectors = fasttext.load_model(model_path)
        embedding_dim = 300
        print("Bulgarian FastText embeddings loaded successfully")

        vocabulary = set()
        for text in texts:
            vocabulary.update(text.split())

        self.word_to_index = {'<PAD>': 0, '<UNK>': 1}
        for i, word in enumerate(sorted(vocabulary), 2):
            self.word_to_index[word] = i

        vocab_size = len(self.word_to_index)
        self.embedding_matrix = torch.zeros(vocab_size, embedding_dim)

        self.embedding_matrix[0] = torch.zeros(embedding_dim)  # [0] => <PAD>
        self.embedding_matrix[1] = torch.randn(embedding_dim) * 0.1  # [1] => <UNK>

        found_words = 0
        for word, index in self.word_to_index.items():
            if word in ['<PAD>', '<UNK>']:
                continue
            try:
                self.embedding_matrix[index] = torch.tensor(words_vectors.get_word_vector(word))
                found_words += 1
            except:
                self.embedding_matrix[index] = torch.randn(embedding_dim) * 0.1
        print(f"Found pre-trained embeddings for {found_words}/{len(self.word_to_index)} words. The other embeddings are randomly initialized.")

        return embedding_dim

    def prepare_data(self) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """
        Load and preprocess training, validation and test data.
        """
        self.data_loader.load_train()
        self.data_loader.load_val()
        self.data_loader.load_test()

        self.authors = self.data_loader.get_author_list()

        train_texts = [self.preprocess_text(text) for text in self.data_loader.train_data['text'].tolist()]
        train_authors = self.data_loader.train_data['author'].tolist()

        val_texts = [self.preprocess_text(text) for text in self.data_loader.val_data['text'].tolist()]
        val_authors = self.data_loader.val_data['author'].tolist()

        test_texts = [self.preprocess_text(text) for text in self.data_loader.test_data['text'].tolist()]
        test_authors = self.data_loader.test_data['author'].tolist()

        all_authors = list(set(train_authors + val_authors + test_authors))
        self.label_encoder.fit(all_authors)

        train_labels = self.label_encoder.transform(train_authors)
        val_labels = self.label_encoder.transform(val_authors)
        test_labels = self.label_encoder.transform(test_authors)

        print(f"Data sizes - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

    def train_model(self, train_texts: List[str], train_labels: List[int], 
                    val_texts: List[str], val_labels: List[int]) -> nn.Module:
        """
        Train the LSTM classifier with the best parameters.

        Parameters:
            train_texts (List[str]): Training text data.
            train_labels (List[int]): Training labels.
            val_texts (List[str]): Validation text data.
            val_labels (List[int]): Validation labels.
        """
        print("Training model with optimized parameters...")

        wandb.init(project="lstm-classifier-optimized", config=self.best_params)

        train_dataset = TextDataset(train_texts, train_labels, self.word_to_index, self.best_params['max_length'])
        train_loader = DataLoader(train_dataset, batch_size=self.best_params['batch_size'], shuffle=True)
        
        val_dataset = TextDataset(val_texts, val_labels, self.word_to_index, self.best_params['max_length'])
        val_loader = DataLoader(val_dataset, batch_size=self.best_params['batch_size'], shuffle=False)

        model = LSTMModel(
            vocab_size=len(self.word_to_index),
            embedding_dim=self.embedding_matrix.size(1),
            hidden_dim=self.best_params['hidden_dim'],
            num_layers=self.best_params['num_layers'],
            num_classes=len(self.authors),
            dropout=self.best_params['dropout'],
            embedding_matrix=self.embedding_matrix
        ).to(self.device)

        # Weighted CrossEntropyLoss
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))

        # Optimizer AdamW
        optimizer = optim.AdamW(model.parameters(), lr=self.best_params['learning_rate'], weight_decay=self.best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_val_acc = 0.0
        patience_counter = 0
        patience = 5

        for epoch in range(self.best_params['epochs']):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_texts, batch_labels in train_loader:
                batch_texts, batch_labels = batch_texts.to(self.device), batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()

            train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0.0
            validation_correct = 0
            validation_total = 0
            
            with torch.no_grad():
                for batch_texts, batch_labels in val_loader:
                    batch_texts, batch_labels = batch_texts.to(self.device), batch_labels.to(self.device)
                    outputs = model(batch_texts)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    validation_total += batch_labels.size(0)
                    validation_correct += (predicted == batch_labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_accuracy = validation_correct / validation_total
            scheduler.step(val_accuracy)

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy
            })

            # Early stopping based on validation accuracy
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        wandb.finish()
        
        return model

    def predict_and_evaluate(self, model: nn.Module, test_texts: List[str], test_labels: List[int]) -> Tuple[float, str]:
        """
        Predict how the texts are classified. 
        Evaluate how good the classification model performs.

        Parameters:
            model (nn.Module): Trained LSTM classification model.
            test_texts (List[str]): Test text data.
            test_labels (List[int]): Test labels.
        """
        test_dataset = TextDataset(test_texts, test_labels, self.word_to_index, self.best_params['max_length'])
        test_loader = DataLoader(test_dataset, batch_size=self.best_params['batch_size'], shuffle=False)

        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_texts, batch_labels in test_loader:
                batch_texts = batch_texts.to(self.device)
                outputs = model(batch_texts)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, 
                                        target_names=self.label_encoder.classes_, zero_division=0)

        wandb.init(project="lstm-classifier-test")
        wandb.log({"test_accuracy": accuracy})

        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

        wandb.finish()

        return accuracy, report


def main():
    os.system("pip install wandb")
    print("Please run '!wandb login' in a Colab cell and paste your W&B API key when prompted.")
    
    # DATA_DIR = 'Recognition-of-authorship-of-text-in-Bulgarian-language/data'
    DATA_DIR = 'data'
    data_loader = TextDataLoader(DATA_DIR)

    lstm_classifier = LSTMClassifier(data_loader)

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = lstm_classifier.prepare_data()

    lstm_classifier.load_pretrained_embeddings(train_texts + val_texts + test_texts)

    model = lstm_classifier.train_model(train_texts, train_labels, val_texts, val_labels)

    accuracy, classification_report = lstm_classifier.predict_and_evaluate(model, test_texts, test_labels)

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(classification_report)

if __name__ == "__main__":
    main()