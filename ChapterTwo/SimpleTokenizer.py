import re


class SimpleTokenizer:

    def __init__(self, raw_text):
        self.raw_tokens = self._tokenize_raw_tokens(raw_text)
        self.vocabulary = self._build_vocabulary()


    def encode(self, text) -> []:
        preprocessed_tokens = self._tokenize_raw_tokens(text)
        encoded_tokens = []
        for token in preprocessed_tokens:
            if token in self.vocabulary:
                encoded_tokens.append(self.vocabulary[token])
            else:
                encoded_tokens.append(self.vocabulary["<|unk|>"])
        return encoded_tokens

    def decode(self, tokens : []) -> []:
        decoded_tokens = []
        for token in tokens:
            for key, value in self.vocabulary.items():
                if token == value:
                    decoded_tokens.append(key)
        return decoded_tokens

    def _tokenize_raw_tokens(self, raw_text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        ## remove the white spaces
        preprocessed = [item.strip() for item in result if item.strip()]
        return preprocessed

    def _build_vocabulary(self):
        unique_tokens = list(sorted(set(self.raw_tokens)))
        vocabulary = {}
        for i, token in enumerate(unique_tokens):
            if token not in vocabulary:
                vocabulary[token] = i

        # We add special tokens to a vocabulary to deal with certain contexts. For instance,
        # we add an <|unk|> token to represent new and unknown words that were not part of the training
        # data and thus not part of the existing vocabulary. Furthermore, we add an <|endoftext|>
        # token that we can use to separate two unrelated text sources.
        vocabulary["<|unk|>"] = len(vocabulary)
        vocabulary["<|endoftext|>"] = len(vocabulary) + 1

        return vocabulary
