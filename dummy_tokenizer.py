class DummyTokenizer:
    def __init__(self):
        self._pad_id = 0
        self._bos_id = 1
        self._eos_id = 2
        self.vocab_size = 20

    def encode(self, text, out_type=int, add_bos=True, add_eos=False):
        tokens = [ord(c) % 17 + 3 for c in text]

        if add_bos:
            tokens = [self._bos_id] + tokens

        if add_eos:
            tokens.append(self._eos_id)

        return tokens

    def decode(self, tokens):
        return " ".join(map(str, tokens))

    def pad_id(self):
        return self._pad_id

    def eos_id(self):
        return self._eos_id