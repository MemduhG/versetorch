from torchtext import data
from torch.autograd import Variable


from src.model.model import subsequent_mask, indexed_bsz_fn


class Batch:
    """Object for holding a batch of data_utils with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MyIterator(data.Iterator):

    def indexed_sort_key(self, sample):
        return self.sort_key(sample[1])

    def create_batches(self):

        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            indexed_data = [(c, x) for c, x in enumerate(self.dataset)]

            self.indices = []
            self.batches = []
            xs = sorted(indexed_data, key=self.indexed_sort_key)
            for b in data.batch(xs, self.batch_size, indexed_bsz_fn):
                sorted_batch = sorted(b, key=lambda x: self.sort_key(x[1]))
                self.batches.append([x[1] for x in sorted_batch])
                self.indices.extend([x[0] for x in sorted_batch])



def rebatch(pad_idx, batch):
    """Fix order in torchtext to match ours"""
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)