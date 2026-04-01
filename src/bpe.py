from dataclasses import dataclass
from nlp_utils import normalize_text, split_sentences, tokenize_sentences
from queue import PriorityQueue


@dataclass
class VocabEntry:
    freq: int
    symbols: list[str]

@dataclass
class MergeRule:
    merge: str
    rank: int

@dataclass(frozen=True) # instância imutável
class TokenPair:
    left: str
    right: str

class bpe_tokenizer:

    def __init__(self, k: int):
        self.vocabulary: dict[str, VocabEntry] = {}
        self.merge_rules: dict[TokenPair, MergeRule] = {}
        self.k = k
    
    def train(self, corpus_text: str) -> None:

        normalized_txt = normalize_text(corpus_text)
        sentences = split_sentences(normalized_txt)
        tokenized_sentences = tokenize_sentences(sentences)

        for words in tokenized_sentences:
            for w in words:
                if w in self.vocabulary:
                    self.vocabulary[w].freq += 1
                else:
                    self.vocabulary[w] = VocabEntry(
                        freq=1,
                        symbols=list(w) + ["</w>"]
                    )

        print("Vocabulary size:",len(self.vocabulary),sep=" ",end="\n\n")

        self.learn()

    def learn(self) -> None:
        for i in range(self.k):
            pairs: dict[TokenPair, int] = {}

            for value in self.vocabulary.values():
                tokens = value.symbols
                freq = value.freq

                for j in range(len(tokens) - 1):
                    p = TokenPair(tokens[j],tokens[j+1])
                    pairs[p] = pairs.get(p,0) + freq

            if not pairs:
                break

            most_frequent_pair = max(pairs, key=lambda k: pairs[k])

            self.merge_rules[most_frequent_pair] = MergeRule(
                merge=most_frequent_pair.left + most_frequent_pair.right,
                rank=i
            )
            self._update_vocab(merge_pair=most_frequent_pair)

    def _update_vocab(self, merge_pair: TokenPair):
        merged_token = merge_pair.left + merge_pair.right

        for value in self.vocabulary.values():
            tokens = value.symbols

            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == (merge_pair.left, merge_pair.right):
                    tokens[i:i+2] = [merged_token] 
                else:
                    i += 1

    def tokenize(self, input: str):
        normalized_input = normalize_text(input)
        word_tokens = tokenize_sentences([normalized_input])[0]
        bpe_rep = []

        for w in word_tokens:
            bpe_rep += list(w) + ["</w>"]

        pairs = PriorityQueue()

        for i in range(len(bpe_rep)-1):
            pair = TokenPair(bpe_rep[i],bpe_rep[i+1])
            rule = self.merge_rules.get(pair)
            if rule is not None:
                rank = rule.rank
                pairs.put((rank,pair))
            
        while not pairs.empty():
            _, best_pair = pairs.get()

            # Verificar se ainda existe
            idx = -1
            for i in range(len(bpe_rep)-1):
                if (bpe_rep[i], bpe_rep[i+1]) == (best_pair.left, best_pair.right):
                    idx = i
                    break
            
            if idx == -1:
                continue
            
            # Fazer merge
            merged = self.merge_rules[best_pair].merge
            bpe_rep[idx:idx+2] = [merged]

            # Atualizar vizinhos
            if idx > 0:
                left_pair = TokenPair(bpe_rep[idx-1], bpe_rep[idx])
                rule = self.merge_rules.get(left_pair)
                if rule:
                    pairs.put((rule.rank, left_pair))
            
            if idx + 1 < len(bpe_rep):
                right_pair = TokenPair(bpe_rep[idx], bpe_rep[idx+1])
                rule = self.merge_rules.get(right_pair)
                if rule:
                    pairs.put((rule.rank, right_pair))

            # i = 0
            # while i < len(bpe_rep)-1:
            #     if (bpe_rep[i],bpe_rep[i+1]) == (best_pair.left, best_pair.right):
            #         bpe_rep[i:i+2] = [self.merge_rules[best_pair].merge]

            #         if i > 0:
            #             left_pair = TokenPair(bpe_rep[i-1],bpe_rep[i])
            #             if left_pair in self.merge_rules:
            #                 pairs.put((self.merge_rules[left_pair].rank, left_pair))
                        
            #         if i + 1 < len(bpe_rep):
            #             right_pair = TokenPair(bpe_rep[i],bpe_rep[i+1])
            #             if right_pair in self.merge_rules:
            #                 pairs.put((self.merge_rules[right_pair].rank, right_pair))   
            #     else:
            #         i += 1

        return bpe_rep

    def print_merge_rules(self):
        print(self.merge_rules)