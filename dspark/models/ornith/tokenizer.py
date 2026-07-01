"""A minimal byte-level tokenizer for demoing/exercising Ornith end-to-end
without depending on an external tokenizer library. Token ids 0-255 map
directly to UTF-8 bytes, so any preset with ``vocab_size >= 256`` works."""


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")
