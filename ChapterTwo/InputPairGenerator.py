import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# the tokenizer used for GPT models also doesn’t use an <|unk|> token
# for out-of-vocabulary words. Instead, GPT models use a byte pair encoding tokenizer,
# which breaks words down into subword units, which we will discuss next.


# ##
# The next step in creating the embeddings for the LLM is to generate the input–target
# pairs required for training an LLM. What do these input–target pairs look like? As we
# already learned, LLMs are pretrained by predicting the next word in a text, as depicted
# in figure 2.12.

class InputPairGenerator(Dataset):

    def __init__(self, raw_text, context_size, stride):
        self.input_ids = []
        self.target_ids = []
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Tokenize the entire text once, up front
        raw_tokens = self.tokenizer.encode(raw_text)

        # Slide a window of length `context_size` across the token sequence.
        # `stride` controls how far the window moves each step:
        #   - stride == 1          -> windows overlap almost completely (max data, max redundancy)
        #   - stride == context_size -> windows don't overlap at all (min redundancy, less data)
        #   - 1 < stride < context_size -> partial overlap, a middle ground
        for index in range(0, len(raw_tokens) - context_size, stride):
            # input_chunk:  tokens [index, index+context_size)
            # target_chunk: same window shifted right by 1 token
            # -> target_chunk[j] is the "next token" for input_chunk[j]
            input_chunk = raw_tokens[index:index + context_size]
            target_chunk = raw_tokens[index + 1:index + context_size + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # DataLoader needs this to know how many samples exist
        return len(self.input_ids)

    def __getitem__(self, idx):
        # DataLoader needs this to fetch one (input, target) pair by index
        return self.input_ids[idx], self.target_ids[idx]
