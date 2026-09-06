import re
import urllib.request

from ChapterTwo.SimpleTokenizer import SimpleTokenizer


## download and break into single words
def download_training_test() -> []:
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "../the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    with open("../the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

raw_text = download_training_test()

tokenizer = SimpleTokenizer(raw_text)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

encoded_text = tokenizer.encode(text)
print(encoded_text)
decoded_tokens = tokenizer.decode(encoded_text)
print(decoded_tokens)
