import sentencepiece as spm
import torchtext as txt

tokenizer_data_files = {"tr": "data/spm/tr.txt"}
tokenizer_model_paths = {"tr": "data/spm/tr.model"}


def get_tokenizer(tokenizer_name: str, vocab_size=32000):
    assert tokenizer_name in tokenizer_model_paths
    sp_model = spm.SentencePieceProcessor()
    try:
        sp_model.Load(tokenizer_model_paths[tokenizer_name])
    except OSError:
        spm_training_string = "--input={} --vocab_size={} --model_prefix={} \
                               --model_type={} --pad_id=3".format(tokenizer_data_files[tokenizer_name], vocab_size,
                                                       "data/spm/" + tokenizer_name, "unigram")
        spm.SentencePieceTrainer.train(spm_training_string)
        sp_model.Load(tokenizer_model_paths[tokenizer_name])
    return sp_model




