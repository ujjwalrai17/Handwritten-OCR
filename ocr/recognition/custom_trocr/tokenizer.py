"""
Character-level tokenizer.
Vocabulary: printable ASCII + 4 special tokens.
  <PAD>=0  <BOS>=1  <EOS>=2  <UNK>=3
"""
import json
import string
from pathlib import Path

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
PAD_ID,    BOS_ID,    EOS_ID,    UNK_ID    = 0, 1, 2, 3


class CharacterTokenizer:
    def __init__(self, extra_chars=None):
        chars = list(
            string.ascii_lowercase + string.ascii_uppercase +
            string.digits + string.punctuation + " "
        )
        if extra_chars:
            chars += [c for c in extra_chars if c not in chars]
        vocab = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(set(chars))
        self.char2id = {ch: i for i, ch in enumerate(vocab)}
        self.id2char = {i: ch for ch, i in self.char2id.items()}

    def encode(self, text: str, add_bos=True, add_eos=True):
        ids = [self.char2id.get(c, UNK_ID) for c in text]
        if add_bos: ids = [BOS_ID] + ids
        if add_eos: ids = ids + [EOS_ID]
        return ids

    def decode(self, token_ids, skip_special=True):
        chars = []
        for tid in token_ids:
            if tid == EOS_ID:
                break
            if skip_special and tid in (PAD_ID, BOS_ID, UNK_ID):
                continue
            chars.append(self.id2char.get(tid, ""))
        return "".join(chars)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.char2id, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path):
        tok = cls.__new__(cls)
        tok.char2id = json.loads(Path(path).read_text())
        tok.id2char = {v: k for k, v in tok.char2id.items()}
        return tok

    @property
    def vocab_size(self):
        return len(self.char2id)

    def __repr__(self):
        return f"CharacterTokenizer(vocab_size={self.vocab_size})"
