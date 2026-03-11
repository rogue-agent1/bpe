#!/usr/bin/env python3
"""Byte Pair Encoding — subword tokenizer used in GPT/LLM models."""
import sys, re
from collections import Counter

class BPE:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size; self.merges = []; self.vocab = {}
    def _get_pairs(self, words):
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    def train(self, text):
        words = Counter(re.findall(r'\w+|[^\w\s]', text.lower()))
        tokenized = {" ".join(w) + " </w>": c for w, c in words.items()}
        base_vocab = set()
        for w in tokenized: base_vocab.update(w.split())
        while len(base_vocab) + len(self.merges) < self.vocab_size:
            pairs = self._get_pairs(tokenized)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_tok = {}
            bigram = " ".join(best)
            replacement = "".join(best)
            for word, freq in tokenized.items():
                new_word = word.replace(bigram, replacement)
                new_tok[new_word] = freq
            tokenized = new_tok
            base_vocab.add(replacement)
        self.vocab = {s: i for i, s in enumerate(sorted(base_vocab | {s for m in self.merges for s in m}))}
    def tokenize(self, text):
        tokens = []
        for word in re.findall(r'\w+|[^\w\s]', text.lower()):
            symbols = list(word) + ["</w>"]
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i+1] == b:
                        symbols[i:i+2] = [a + b]
                    else:
                        i += 1
            tokens.extend(symbols)
        return tokens
    def encode(self, text):
        return [self.vocab.get(t, -1) for t in self.tokenize(text)]

if __name__ == "__main__":
    corpus = """The quick brown fox jumps over the lazy dog. The dog barked at the fox.
    Natural language processing uses tokenization. Byte pair encoding is a subword tokenizer.
    Machine learning models process tokens not words. GPT uses BPE tokenization."""
    bpe = BPE(vocab_size=200)
    bpe.train(corpus)
    print(f"Vocab size: {len(bpe.vocab)}, Merges: {len(bpe.merges)}")
    print(f"\nTop merges: {bpe.merges[:10]}")
    test = sys.argv[1] if len(sys.argv) > 1 else "tokenization"
    tokens = bpe.tokenize(test)
    ids = bpe.encode(test)
    print(f"\nInput: {test!r}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
