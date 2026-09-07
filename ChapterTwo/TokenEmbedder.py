import torch


class TokenEmbedder:

    def __init__(self, raw_text, data_loader, vocab_size, output_dim, context_length):
        self.raw_text = raw_text
        self.data_loader = data_loader
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        self.pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
        self.pos_embeddings = self.pos_embedding_layer(torch.arange(context_length))

    def embed_positionally(self, inputs):
        return self.token_embedding_layer(inputs) + self.pos_embeddings

