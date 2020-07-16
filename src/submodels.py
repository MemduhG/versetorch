import torch
from torch import nn
from posn import PositionalEncoding
from utils import get_tokenizer


class TransformerModel(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, vocab_size=32000, tokenizer="tr", batch_size=32):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.config_string = "-".join(str(x) for x in[d_model, nhead, num_encoder_layers, num_decoder_layers,
                                                      dim_feedforward, dropout, vocab_size, tokenizer])
        self.posn = PositionalEncoding(d_model=d_model)
        self.start_symbol, self.end_symbol, self.pad_token = 1, 2, 3
        self.embedding = nn.Embedding(vocab_size + 2, d_model)
        self.vocab_out = nn.Linear(d_model, vocab_size)
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer(tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, src, tgt, src_key_mask=None, tgt_key_mask=None):
        src_embedded, tgt_embedded = self.embedding(src), self.embedding(tgt)
        transformer_out = self.transformer(src_embedded, tgt_embedded, src_key_padding_mask=src_key_mask,
                                           tgt_key_padding_mask=tgt_key_mask)
        tokens_out = self.vocab_out(transformer_out)
        return tokens_out

    def tokenize_string(self, src, out_type=int):
        return self.tokenizer.encode(src, out_type=out_type)

    def decode_string(self, src_indices):
        return self.tokenizer.decode(src_indices)

    def predict_inference_src(self, src: str, max_len: int = 512) -> str:
        """
        Used on inference to predict
        :param src: String text as source
        :param max_len: max length of the generator
        :return: predicted text
        """
        src_input = torch.LongTensor(self.tokenize_string(src))

        src_tensor = self.embedding(torch.LongTensor(src_input)).unsqueeze(1)

        memory = self.transformer.encoder(src_tensor)

        tgt_input = [self.start_symbol]

        for i in range(max_len):
            tgt_tensor = self.embedding(torch.LongTensor(tgt_input)).unsqueeze(1)

            decoder_out = self.transformer.decoder(tgt_tensor, memory)
            tokens_out = self.vocab_out(decoder_out)
            out_token = tokens_out.argmax(2)[-1].item()
            tgt_input.append(out_token)
            if out_token == self.end_symbol:
                break

        return self.decode_string(tgt_input)

    def batch_generator(self, text_file):
        with open(text_file, encoding="utf-8") as infile:
            lines = [x.rstrip("\n") for x in infile.readlines()]
        for i in range(0, len(lines), self.batch_size):
            tokenized = [self.tokenize_string(x) for x in lines[i:i+self.batch_size]]
            tensorized = [torch.LongTensor(x) for x in tokenized]
            tensor = torch.nn.utils.rnn.pad_sequence(tensorized, batch_first=True, padding_value=3)
            masks = [[False for _ in range(len(sentence))] + [True for _ in range(tensor.shape[1] - len(sentence))]
                     for sentence in tokenized]
            yield torch.reshape(tensor, (tensor.shape[1], tensor.shape[0])).to(self.device), \
                torch.BoolTensor(masks).to(self.device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    model = TransformerModel(d_model=128, dim_feedforward=128, num_decoder_layers=2, num_encoder_layers=2)
    model.load("iter700.pt")
    decoded = model.predict_inference_src("Ekmek aldım, eve gittim.", max_len=128)
    print(decoded)