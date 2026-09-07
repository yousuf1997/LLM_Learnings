# the tokenizer used for GPT models also doesn’t use an <|unk|> token
# for out-of-vocabulary words. Instead, GPT models use a byte pair encoding tokenizer,
# which breaks words down into subword units, which we will discuss next.
from ChapterTwo.InputPairGenerator import InputPairGenerator

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

InputPairGenerator(raw_text, 4, 4)

#48


