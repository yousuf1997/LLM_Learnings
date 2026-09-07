import urllib

import torch

from ChapterTwo.DataLoaderHelper import DataLoaderHelper
from ChapterTwo.TokenizerExample import data_loader

def download_training_test():
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

vocab_size = 50257
output_dim = 256
context_size = 4
batch_size = 8
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

data_loader_helper = DataLoaderHelper()

data_loader = data_loader_helper.create_data_loader(
     download_training_test(),
     context_size,
     batch_size,
     context_size,
     False
)

data_iter = iter(data_loader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# Let’s now use the embedding layer to embed these token IDs into 256-dimensional
# vectors:

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
