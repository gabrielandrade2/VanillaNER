import glob
from abc import ABC, abstractmethod

import pandas as pd


def get_file_list(directory: str, extension: str):
    if not directory[-1] == '/':
        directory += '/'
    return glob.glob(directory + '[!~]*.' + extension)


def load_from_xls_file(file):
    pass


class Dataset(ABC):
    '''
    Class that represents a simple collection of texts that can be iterated as a list.
    It encapsulates the code needed to open sets of texts stored in different formats, e.g. single file, folder,
    collection of multiple files with multiple texts inside it, Excel files, etc.
    '''

    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        text = None
        while not text:
            text = self._next_text()
            if text != text:
                continue
            return text

    def __len__(self):
        return self._length()

    @abstractmethod
    def _next_text(self):
        '''
        Abstract method to be implemented in extending classes.
        This should return the next text of the dataset and must raise StopIteration when all texts were returned.

        :return: The next text
        :raises StopIteration when there are no more texts to return.
        '''
        raise StopIteration

    @abstractmethod
    def _length(self):
        '''
        Abstract method to be implemented in extending classes.
        This should return the number of texts that compose the dataset. The way it is calculated can be different
        depending on the type of dataset.

        :return: The number of texts in the dataset.
        '''
        raise NotImplemented

    def as_list(self):
        '''
        Returns a list containing all the elements of the current dataset.

        :return: A list containing all the elements of the current dataset.
        '''
        return [text for text in self]


class TwitterDataset(Dataset):

    def __init__(self, directory):
        self.file_list = get_file_list(directory, 'csv')
        self.current_texts = None
        self.file_num = 0
        self.text_num = 0
        self.length = None

    def __iter__(self):
        self.file_num = 0
        self.text_num = 0
        self.open_next_file()
        return self

    def open_next_file(self):
        if self.file_num >= len(self.file_list):
            raise StopIteration

        file = self.file_list[self.file_num]
        # print('\nFile', self.file_num + 1, 'of', len(self.file_list))
        # print(file)
        csv = pd.read_csv(file, sep='^([^,]+),', engine='python', header=None)
        # Get relevant columns
        self.current_texts = csv[2].to_list()

        self.file_num += 1
        self.text_num = 0

    def _next_text(self):
        if self.text_num >= len(self.current_texts):
            self.open_next_file()

        # print('Text', self.text_num + 1, 'of', len(self.current_texts), end='\r')
        text = self.current_texts[self.text_num]
        self.text_num += 1
        return text

    def _length(self):
        if not self.length:
            self.length = 0
            for file in self.file_list:
                self.length += sum(1 for line in open(file, 'r'))

        return self.length


class YakurekiTxtDataset(Dataset):
    def __init__(self, directory):
        self.file_list = get_file_list(directory, 'txt')
        self.current_texts = None
        self.file_num = 0

    def __iter__(self):
        self.file_num = 0
        return self

    def _next_text(self):
        if self.file_num >= len(self.file_list):
            raise StopIteration

        file = self.file_list[self.file_num]
        # print(file, '\n', 'File',self.file_num + 1, 'of', len(self.file_list), end='\r')
        self.file_num += 1

        file = open(file, 'r')

        # Ignore metadata in first line
        # ['ID', 'Drug', 'Adverse Event', 'Place'])  # ID,  薬剤名, 有害事象, 想定した服薬指導実施場所
        return ''.join(file.readlines()[1:])

    def _length(self):
        return len(self.file_list)
