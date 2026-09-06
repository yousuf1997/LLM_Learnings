from torch.utils.data import DataLoader

from ChapterTwo.InputPairGenerator import InputPairGenerator

class DataLoaderHelper:

    def __init__(self):
        pass

    def create_data_loader(self, text, context_size, batch_size, stride, shuffle=True, drop_last=True, num_workers=0) :
        data_pairs = InputPairGenerator(text, context_size, stride)

        return DataLoader(data_pairs, batch_size=batch_size, shuffle=shuffle,  drop_last=drop_last, num_workers=num_workers)
