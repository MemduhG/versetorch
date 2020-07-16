import sentencepiece as spm
import sacrebleu

tokenizer_data_files = {"tr": "data/spm/tr.txt"}
tokenizer_model_paths = {"tr": "data/spm/tr.model"}

data_paths = {"antoloji": {"src": "data/tr/antoloji.train.prose", "tgt": "data/tr/antoloji.train.poetry",
                           "dev_src": "data/tr/antoloji.dev.prose", "dev_tgt": "data/tr/antoloji.dev.poetry"}}


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


def score_translation(system_output, reference):
    return sacrebleu.corpus_bleu(system_output, reference)



