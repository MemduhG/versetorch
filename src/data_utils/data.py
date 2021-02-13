from src.model.model import batch_size_fn
from src.data_utils.batch import MyIterator
from torchtext import data, datasets
from src.utils.utils import get_tokenizer

import torch


def each_line(fname):
    lines = []
    with open(fname, "r", encoding="utf-8") as infile:
        for line in infile:
            lines.append(line.strip())
    return lines


def get_dataset_strings(dataset):
    languages = {"antoloji": "tr", "tur": "tr"}
    language = languages[dataset]
    train = each_line('data/{}/{}.{}.src'.format(language, dataset, "train"))
    dev = each_line('data/{}/{}.{}.src'.format(language, dataset, "dev"))
    test = each_line('data/{}/{}.{}.src'.format(language, dataset, "dev"))

    return train, dev, test


def tokenize_string(string, tokenizer):
    return [1] + tokenizer.EncodeAsIds(string) + [2]


def get_tokenized_dataset(dataset):
    train, dev, test = get_dataset_strings(dataset)
    languages = {"antoloji": "tr", "tur": "tr"}
    language = languages[dataset]
    tokenizer = get_tokenizer(language)
    train_set = [tokenize_string(x, tokenizer) for x in train]
    dev_set = [tokenize_string(x, tokenizer) for x in dev]
    test_set = [tokenize_string(x, tokenizer) for x in test]
    return train_set, dev_set, test_set

def produce_register(dataset):
    train, dev, test = get_tokenized_dataset(dataset)
    train_indices = {tuple(item): c for c, item in enumerate(train)}
    dev_indices = {tuple(item): c for c, item in enumerate(dev)}
    test_indices = {tuple(item): c for c, item in enumerate(test)}
    return train_indices, dev_indices, test_indices


def get_dataset(dataset):
    languages = {"antoloji": "tr", "tur":"tr", "turkish":"tr"}
    language = languages[dataset]
    tokenizer = get_tokenizer(language)

    def tok(seq):
        return tokenizer.EncodeAsIds(seq)
    src = data.Field(tokenize=tok, init_token=1, eos_token=2, pad_token=3, use_vocab=False)
    tgt = data.Field(tokenize=tok, init_token=1, eos_token=2, pad_token=3, use_vocab=False)
    mt_train = datasets.TranslationDataset(
        path='data/{}/{}.train'.format(language, dataset), exts=('.src', '.tgt'),
        fields=(src, tgt))
    mt_dev = datasets.TranslationDataset(
        path='data/{}/{}.dev'.format(language, dataset), exts=('.src', '.tgt'),
        fields=(src, tgt))
    mt_test = datasets.TranslationDataset(
        path='data/{}/{}.test'.format(language, dataset), exts=('.src', '.tgt'),
        fields=(src, tgt))
    return mt_train, mt_dev, mt_test


def get_training_iterators(dataset):
    if torch.cuda.device_count() > 1:
        batch_size = 8000
    else:
        batch_size = 1000
    train, val, test = get_dataset(dataset)

    train_iter = MyIterator(train, batch_size=batch_size, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=batch_size, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False, sort=True)
    test_iter = MyIterator(test, batch_size=batch_size, device=0,
                           repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                           batch_size_fn=batch_size_fn, train=False, sort=True)

    train_idx, dev_idx, test_idx = produce_register(dataset)

    return train_iter, valid_iter, test_iter, train_idx, dev_idx, test_idx


if __name__ == "__main__":
    train, val, test = get_training_iterators("tur")
    for batch in val:
        print(batch)
        break

