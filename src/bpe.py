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


class bpe_tokenizer:

    def __init__(self, text:str, k: int):
        self.vocabulary: dict[str, VocabEntry] = {}
        self.merge_rules: dict[tuple[str,str], MergeRule] = {}
        self.k = k

        normalized_txt = normalize_text(text)
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

    def learn(self) -> None:
        for i in range(self.k):
            pairs = {}
            for value in self.vocabulary.values():
                tokens = value.symbols
                freq = value.freq
                number_of_tokens = len(tokens)
                if number_of_tokens > 1:
                    for j in range(1,number_of_tokens):
                        p = (tokens[j-1],tokens[j])
                        pairs[p] = pairs.get(p,0) + freq

            most_frequent_pair = max(pairs,key=pairs.get)
            self.merge_rules[most_frequent_pair] = MergeRule(
                merge=''.join(most_frequent_pair),
                rank=i
            )
            self._update_vocab(merge_pair=most_frequent_pair)

    def _update_vocab(self,merge_pair:tuple[str,str]):
        merged_token = ''.join(merge_pair)

        for value in self.vocabulary.values():
            tokens = value.symbols

            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == merge_pair:
                    tokens[i:i+2] = [merged_token] 
                else:
                    i += 1

    def tokenize(self,input: str):
        normalized_input = normalize_text(input)
        word_tokens = tokenize_sentences([normalized_input])[0]
        bpe_rep = []
        for w in word_tokens:
            bpe_rep += list(w) + ["</w>"]

        pairs = PriorityQueue()

        for i in range(len(bpe_rep)-1):
            pair = (bpe_rep[i],bpe_rep[i+1])
            rule = self.merge_rules.get(pair)
            if rule is not None:
                rank = rule.rank
                pairs.put((rank,pair))
            
        while not pairs.empty():
            best_pair = pairs.get()[1]
            i = 0
            while i < len(bpe_rep)-1:
                if (bpe_rep[i],bpe_rep[i+1]) == best_pair:
                    bpe_rep[i:i+2] = [self.merge_rules[best_pair].merge]
                    left_idx = i - 1
                    right_idx = i + 1

                    if left_idx >= 0:
                        left_pair = (bpe_rep[left_idx],bpe_rep[i])
                        rule = self.merge_rules.get(left_pair)
                        if rule is not None:
                            rank = rule.rank
                            pairs.put((rank,left_pair))
                        
                    if right_idx < len(bpe_rep):
                        right_pair = (bpe_rep[i],bpe_rep[right_idx])
                        rule = self.merge_rules.get(right_pair)
                        if rule is not None:
                            rank = rule.rank
                            pairs.put((rank,right_pair))   
                i += 1
        return bpe_rep

    def print_merge_rules(self):
        print(self.merge_rules)
            