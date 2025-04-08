import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

def preprocess_data(df):
    df = df[['features', 'description']].dropna()
    df['input_text'] = "generate description: " + df['features']
    df['target_text'] = df['description']
    return df

def tokenize(batch, tokenizer, max_input_length=256, max_target_length=128):
    inputs = tokenizer(batch['input_text'], padding="max_length", truncation=True, max_length=max_input_length)
    targets = tokenizer(batch['target_text'], padding="max_length", truncation=True, max_length=max_target_length)
    inputs['labels'] = targets['input_ids']
    return inputs

def train_model(data_path='data/products.csv'):
    print("Starting model training...")

    # Load and preprocess data
    df = pd.read_csv(data_path)
    df = preprocess_data(df)

    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')  # This is enough to load the model with weights

    # Tokenize data
    from datasets import Dataset
    dataset = Dataset.from_pandas(df[['input_text', 'target_text']])
    tokenized_dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)

    # Define training arguments
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs',
        save_strategy="no",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    os.makedirs("app/model", exist_ok=True)
    model.save_pretrained("app/model")
    tokenizer.save_pretrained("app/model")

    print("Model saved successfully to 'app/model'")
    return model

# Run the training process
train_model()