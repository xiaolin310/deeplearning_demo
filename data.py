
import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.strip().split(",")
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.strip().split(',')
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                #print(ids)
                idss.append(torch.tensor(ids).type(torch.float))
            ids = torch.cat(idss)
        # print(len(ids))
        # print(len(ids) % 9)
        return ids
    
if __name__ == '__main__':
    corpus = Corpus("./data")
    print(corpus.dictionary.word2idx)
    # print(corpus.dictionary.idx2word)

    print(corpus.train)
    print(corpus.train.shape)
    print(len(corpus.dictionary))

