from model import batch_size_fn
from batch import MyIterator
from torchtext import data, datasets
from utils import get_tokenizer

import torch


def get_dataset(dataset):
    languages = {"antoloji": "tr"}
    language = languages[dataset]
    tokenizer = get_tokenizer(language)

    def tok(seq):
        return tokenizer.EncodeAsIds(seq)
    src = data.Field(tokenize=tok, init_token=1, eos_token=2, pad_token=3, use_vocab=False)
    tgt = data.Field(tokenize=tok, init_token=1, eos_token=2, pad_token=3, use_vocab=False)
    mt_train = datasets.TranslationDataset(
        path='data/{}/{}.train'.format(language, dataset), exts=('.prose', '.poetry'),
        fields=(src, tgt))
    mt_dev = datasets.TranslationDataset(
        path='data/{}/{}.dev'.format(language, dataset), exts=('.prose', '.poetry'),
        fields=(src, tgt))
    mt_test = datasets.TranslationDataset(
        path='data/{}/{}.dev'.format(language, dataset), exts=('.prose', '.poetry'),
        fields=(src, tgt))
    return mt_train, mt_dev, mt_test


def get_training_iterators(dataset):
    if torch.cuda.device_count() > 1:
        batch_size = 12000
    else:
        batch_size = 1000
    train, val, test = get_dataset(dataset)

    train_iter = MyIterator(train, batch_size=batch_size, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=batch_size, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    test_iter = MyIterator(test, batch_size=batch_size, device=0,
                           repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                           batch_size_fn=batch_size_fn, train=False)

    return train_iter, valid_iter, test_iter


if __name__ == "__main__":
    pass
