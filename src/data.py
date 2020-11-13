from torchtext.data.functional import load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer
from torchtext import data, datasets
from utils import get_tokenizer
BLANK_WORD = "<blank>"
PAD_IDX = 3


def get_dataset(dataset):
    tokenizer = get_tokenizer("tr")
    def tok(seq):
        return tokenizer.EncodeAsIds(seq)
    SRC = data.Field(tokenize=tok, init_token=1, eos_token=2, pad_token=3, use_vocab=False)
    TGT = data.Field(tokenize=tok, init_token=1, eos_token=2, pad_token=3, use_vocab=False)
    mt_train = datasets.TranslationDataset(
        path='data/tr/antoloji.train', exts=('.prose', '.poetry'),
        fields=(SRC, TGT))
    mt_dev = datasets.TranslationDataset(
        path='data/tr/antoloji.dev', exts=('.prose', '.poetry'),
        fields=(SRC, TGT))
    mt_test = datasets.TranslationDataset(
        path='data/tr/antoloji.dev', exts=('.prose', '.poetry'),
        fields=(SRC, TGT))
    return SRC, TGT, mt_train, mt_dev, mt_test


if __name__ == "__main__":
    tokenizer = get_tokenizer("tr")
    st = "Yar misalini ne zemin u zaman gormustur"
    a = tokenizer.EncodeAsPieces(st)
    print(a)