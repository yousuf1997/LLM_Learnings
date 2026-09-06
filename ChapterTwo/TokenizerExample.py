import re
import urllib.request

from ChapterTwo.DataLoaderHelper import DataLoaderHelper
from ChapterTwo.SimpleTokenizer import SimpleTokenizer


## download and break into single words
def download_training_test() -> []:
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

raw_text = download_training_test()


data_loader = DataLoaderHelper()

data_set = data_loader.create_data_loader(
    raw_text,
    4,
    1,
    1,
    False
)

data_iter = iter(data_set)
first_batch = next(data_iter)
print(first_batch)
