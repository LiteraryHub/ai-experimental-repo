import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import os


def read_json_file(file_path):
    try:
        # Attempt to read the file assuming it's line-delimited or an array of objects
        data = pd.read_json(file_path, lines=True)
    except ValueError:
        try:
            # For a single JSON object, read it into a Series and then convert to a DataFrame
            data = pd.read_json(file_path, typ='series')
            data = pd.DataFrame([data])  # Convert Series to DataFrame
        except ValueError as e:
            print(f"Failed to read {file_path}: {e}")
            data = pd.DataFrame()  # Return an empty DataFrame on failure
    return data


def read_json_files(directory):
    files = glob.glob(f"{directory}\\*.json")
    data_frames = [read_json_file(file) for file in files]
    # Filter out empty DataFrames
    data_frames = [df for df in data_frames if not df.empty]
    if data_frames:
        # Concatenate all DataFrames
        return pd.concat(data_frames, ignore_index=True)
    else:
        print("No valid JSON data could be loaded.")
        return pd.DataFrame()

# Preprocess and tokenize the text
def tokenize_function(examples, tokenizer):
    print(type(examples["text_chunk"].values))
    return tokenizer(examples["text_chunk"].values, padding="max_length", truncation=True, max_length=512)


# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    # Get current dir 
    directory = os.getcwd() + "\\src\\nlp\\restricted_topic_detection\\labeled_dataset"
    df = read_json_files(directory)
    print(df.head())
    # Split the dataset
    train_df, eval_df = train_test_split(df, test_size=0.2)

    # Initialize tokenizer
    model_name = "aubmindlab/bert-base-arabertv02"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize data
    tokenized_train = tokenize_function(train_df, tokenizer)
    tokenized_eval = tokenize_function(eval_df, tokenizer)

    # Prepare datasets
    train_dataset = CustomDataset(tokenized_train, train_df["label"].tolist())
    eval_dataset = CustomDataset(tokenized_eval, eval_df["label"].tolist())

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        evaluate_during_training=True,
        logging_dir="./logs",
    )

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Train and evaluate
    trainer.train()
    trainer.evaluate()
    
    # I want to saved 

    # Save model and tokenizer
    model.save_pretrained("./src/nlp/restricted_topic_detection/fine_tuned_AraBERT_model")
    tokenizer.save_pretrained("./src/nlp/restricted_topic_detection/fine_tuned_AraBERT_tokenizer")
