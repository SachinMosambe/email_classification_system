import pandas as pd
# Add this import if you want to use display() in Jupyter or IPython environments


try:
    df_emails = pd.read_csv('combined_emails_with_natural_pii - combined_emails_with_natural_pii.csv', encoding='latin-1')

except FileNotFoundError:
    print("Error: 'combined_emails_with_natural_pii - combined_emails_with_natural_pii.csv' not found.")
    df_emails = None
except pd.errors.ParserError:
    print("Error: Could not parse the CSV file. Please check the file format.")
    df_emails = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df_emails = None



"""## Data preparation

### Subtask:
Prepare the data for PII masking.

**Reasoning**:
Create a copy of the dataframe and add the original email text column.
"""

# Assuming the email text is in the 'email' column
if df_emails is not None:
    df_masked_emails = df_emails.copy()
    df_masked_emails['original_email_text'] = df_masked_emails['email'].copy()
else:
    print("The dataframe df_emails is not properly loaded, please check previous steps.")

"""## Data cleaning

### Subtask:
Mask Personally Identifiable Information (PII) in the email text.

"""

import re

def mask_email(text):
    """Masks email addresses in the input text."""
    masked_text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[email]", text)
    return masked_text

def mask_phone_number(text):
    """Masks phone numbers in the input text."""
    masked_text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[phone_number]", text)
    return masked_text

def mask_full_name(text):
    """Masks full names in the input text. This is a basic example and might need improvement."""
    masked_text = re.sub(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", "[full_name]", text)
    return masked_text

# Placeholder functions for other PII types
def mask_dob(text):
    return re.sub(r"\d{2}/\d{2}/\d{4}", "[dob]", text)

def mask_aadhar_num(text):
    return re.sub(r"\d{12}", "[aadhar_num]", text)

def mask_credit_debit_no(text):
    return re.sub(r"\d{16}", "[credit_debit_no]", text)

def mask_cvv_no(text):
    return re.sub(r"\d{3,4}", "[cvv_no]", text)

def mask_expiry_no(text):
    return re.sub(r"\d{2}/\d{2}", "[expiry_no]", text)


if df_masked_emails is not None:
    df_masked_emails['masked_email'] = df_masked_emails['email'].apply(mask_email)
    df_masked_emails['masked_email'] = df_masked_emails['masked_email'].apply(mask_phone_number)
    df_masked_emails['masked_email'] = df_masked_emails['masked_email'].apply(mask_full_name)
    df_masked_emails['masked_email'] = df_masked_emails['masked_email'].apply(mask_dob)
    df_masked_emails['masked_email'] = df_masked_emails['masked_email'].apply(mask_aadhar_num)
    df_masked_emails['masked_email'] = df_masked_emails['masked_email'].apply(mask_credit_debit_no)
    df_masked_emails['masked_email'] = df_masked_emails['masked_email'].apply(mask_cvv_no)
    df_masked_emails['masked_email'] = df_masked_emails['masked_email'].apply(mask_expiry_no)

    # Security documentation
    print("Security Measures for Original Email Data:")
    print("- The 'original_email_text' column in the 'df_masked_emails' DataFrame holds the original, unmasked email data.")
    print("- This data should be stored securely, ideally in an encrypted format or within a restricted-access database.")
    print("- Access to this data should be limited to authorized personnel only.")
    print("- Regular security audits should be conducted to ensure the ongoing protection of this sensitive information.")
    print("- Implement robust access control mechanisms to prevent unauthorized access to the storage location.")
else:
    print("The dataframe df_masked_emails is not properly loaded, please check previous steps.")

"""## Data wrangling

### Subtask:
Prepare the masked email data for model training.

"""

from transformers import RobertaTokenizer
import torch

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  # Use a pre-trained RoBERTa tokenizer

# Tokenization and label encoding
max_length = 512  # Define maximum sequence length

input_ids = []
attention_masks = []
labels = []
label_map = {}  # Create label mapping

unique_labels = df_masked_emails["type"].unique()
for i, label in enumerate(unique_labels):
    label_map[label] = i

for index, row in df_masked_emails.iterrows():
    encoded_dict = tokenizer.encode_plus(
        row["masked_email"],
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    labels.append(label_map[row["type"]])

# Convert lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Create the training dataset
training_data = {
    "input_ids": input_ids,
    "attention_mask": attention_masks,
    "labels": labels
}
print(f"Label Mapping: {label_map}")

for key, value in training_data.items():
  print(f"Key: {key}, Shape: {value.shape}")

import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Get the indices for splitting
train_indices, val_test_indices = train_test_split(
    np.arange(len(training_data["labels"])), test_size=0.2, random_state=42, stratify=training_data["labels"].numpy()
)
val_indices, test_indices = train_test_split(
    val_test_indices, test_size=0.5, random_state=42, stratify=training_data["labels"][val_test_indices].numpy()
)

# Create datasets by indexing the tensors in training_data
train_dataset = {
    key: value[train_indices] for key, value in training_data.items()
}
val_dataset = {
    key: value[val_indices] for key, value in training_data.items()
}
test_dataset = {
    key: value[test_indices] for key, value in training_data.items()
}

# Print shapes of 'input_ids' tensors to verify
print(f"Training set input_ids shape: {train_dataset['input_ids'].shape}")
print(f"Validation set input_ids shape: {val_dataset['input_ids'].shape}")
print(f"Test set input_ids shape: {test_dataset['input_ids'].shape}")

"""## Model training

### Subtask:
Train a RoBERTa model for sequence classification.

**Reasoning**:
Train the RoBERTa model using the prepared datasets.
"""

print(type(train_dataset))
print(train_dataset.keys())
for key, value in train_dataset.items():
    print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape}")
    print(value[0])

"""## Model training

### Subtask:
Train a RoBERTa model for sequence classification.

"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
import numpy as np

class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Prepare full datasets
train_encodings = {
    "input_ids": train_dataset["input_ids"].numpy(),
    "attention_mask": train_dataset["attention_mask"].numpy()
}
train_labels = train_dataset["labels"].numpy()
train_dataset_full = EmailDataset(train_encodings, train_labels)

val_encodings = {
    "input_ids": val_dataset["input_ids"].numpy(),
    "attention_mask": val_dataset["attention_mask"].numpy()
}
val_labels = val_dataset["labels"].numpy()
val_dataset_full = EmailDataset(val_encodings, val_labels)

# Load the pre-trained model and tokenizer
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_map))
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_roberta",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_steps=100,
    save_strategy="epoch",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_full,
    eval_dataset=val_dataset_full,
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./fine_tuned_roberta")
tokenizer.save_pretrained("./fine_tuned_roberta")



# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

# Inference
def predict_email_type(email_text, model, tokenizer, label_map):
    """Predicts the type of an email using the fine-tuned RoBERTa model."""
    inputs = tokenizer(email_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = inputs.to(device)
    outputs = model(**inputs)
    predicted_label_id = torch.argmax(outputs.logits).item()

    # Reverse the label mapping to get the original label
    reverse_label_map = {v: k for k, v in label_map.items()}
    predicted_label = reverse_label_map[predicted_label_id]

    return predicted_label

# Example usage
example_email = "Subject: Problem Regarding the Launch of Digital Marketing Campaigns,Dear Customer Support, I am contacting you to report an issue with our digital marketing campaigns. The campaigns have not been able to launch, and we believe that there may be technical issues with the integrations. Despite attempting to restart the systems and checking the connections, the problem continues. We would be grateful if you could investigate this and provide a resolution as quickly as possible. Please inform us if you require any further details from our side You can reach me at johndoe@email.com.. We are eagerly awaiting your response.. Warm regards, Elena Ivanova"

predicted_type = predict_email_type(example_email, model, tokenizer, label_map)
print(f"Predicted Email Type: {predicted_type}")








