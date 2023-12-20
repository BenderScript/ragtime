from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Text for testing
text = "Hello, my name is John."

# Tokenize input
tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Predict hidden states features for each layer
with torch.no_grad():
    outputs = model(**tokens)

# Get the embeddings from the last BERT layer
last_layer_embeddings = outputs.last_hidden_state

print("Tokens:", tokens)
print("\nLast layer embeddings:", last_layer_embeddings)
