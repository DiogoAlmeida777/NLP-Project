from collections import Counter
import re
import math
import random
from nlp_utils import normalize_text, split_sentences, tokenize_sentences
from bpe import bpe_tokenizer



class NGram:
    def __init__(self,N: int,lambda_backoff:float = 0.4,tokenizer:bpe_tokenizer | None = None):
        self.N: int = N
        #self.ngram_counts: dict[int,dict[tuple[str, ...],int]] = dict()
        #self.vocab: set[str] = set()   
        #self.total_tokens: int = 0 
        self.lambda_backoff: float = lambda_backoff
        self.eps: float = 10**-12
        self.tokenizer = tokenizer

        
    def train(self,corpus_text: str) -> None:
        self.ngram_counts = {
            n: {} for n in range(self.N)
        }
        self.vocab: set[str] = set() 
        self.total_tokens: int = 0     

        # creates a tokenizer if it doesn't exist yet.
        if self.tokenizer is None:
            self.tokenizer = bpe_tokenizer(corpus_text,5000)
        
        # makes the tokenizer learn if it doesn't have any merge rule yet.
        if not self.tokenizer.merge_rules:
            self.tokenizer.learn()

        # Pre-Processing
        tokenized_sentences = self._process_text(corpus_text)

        for tokens in tokenized_sentences:
            for i, t in enumerate(tokens):
                unigram = (t,)
                self.ngram_counts[0][unigram] = self.ngram_counts[0].get(unigram,0) + 1
                end = min(self.N,i+1) 
                for j in range(1,end):
                    key = tuple(tokens[i-j:i+1])
                    self.ngram_counts[j][key] = self.ngram_counts[j].get(key,0) + 1

                if t != '<s>':
                    self.vocab.add(t)
                    self.total_tokens += 1
            

    def _stupid_backoff(self,ngram:tuple[str,...]):
        i = self.N - 1
        power = 0
        probability = 0

        while i > 0:
            ngram_count = self.ngram_counts[i].get(ngram,0)

            if ngram_count > 0:
                previous_words = ngram[:-1]
                previous_words_count = self.ngram_counts[i-1][previous_words]
                probability = (ngram_count/previous_words_count) * (self.lambda_backoff**power)
                return probability
            
            ngram = ngram[1:]
            i -= 1
            power += 1
        
        unigram_count = self.ngram_counts[0].get(ngram,0)
        probability = unigram_count / self.total_tokens
        return probability * (self.lambda_backoff**power)
            
            
    def sentence_score(self,sentence: str) -> float:
        tokens = self._process_text(sentence)[0]
        sum_log_scores = 0
        for i, _ in enumerate(tokens):
            if i > self.N - 2:
                ngram = tuple(tokens[i - (self.N-1):i+1])
                score = self._stupid_backoff(ngram)
                sum_log_scores += math.log(score + self.eps)
        
        return math.exp(sum_log_scores)


    def generate_sentence(self,max_len: int = 30, temperature: float = 1.0) -> str:
        ctx = ['<s>'] * (self.N - 1)

        for i in range(max_len):
            candidate_exps = {}
            sum_of_exps = 0
            candidate_proba = {}
            for candidate in self.vocab: 
                # get the last n-1 tokens from ctx
                n_tokens = ctx[-(self.N-1):]
                # add the candidate
                n_tokens.append(candidate)
                # create the ngram with the n tokens
                ngram = tuple(n_tokens)
                score = self._stupid_backoff(ngram)
                logit = math.log(score + self.eps)
                exp = math.exp(logit/temperature)
                candidate_exps[candidate] = exp
                sum_of_exps += exp

            for candidate, exp in candidate_exps.items():
                softmax = exp / sum_of_exps
                candidate_proba[candidate] = softmax
            
            rng = random.random()
            sum_of_probas = 0
            for candidate, proba in candidate_proba.items():
                sum_of_probas += proba
                if rng < sum_of_probas:
                    ctx.append(candidate)
                    break
            
            if ctx[-1] == '</s>':
                break

        generated_sentence = ''.join(ctx)
        # substitutes the </w> with a whitespace (if they are not followed by punctuation)
        # ?! - negative lookahead, it only matches </w> if it is not followed by punctuation.
        generated_sentence = re.sub(r'</w>(?![`.,!?;:()\"\'’-])',' ',generated_sentence)
        # removes any remaining </w> and all the <s> and </s>.
        return re.sub(r'</w>|<s>|</s>','',generated_sentence)



    def perplexity(self,text: str) -> float:
        tokenized_sentences = self._process_text(text)
        sum_of_logs = 0
        n = 0
        for tokens in tokenized_sentences:
            for i in range((self.N-1),len(tokens)):
                ngram = tuple(tokens[i-(self.N-1):i+1])
                score = self._stupid_backoff(ngram)
                sum_of_logs += math.log(score + self.eps)
                n += 1
        return math.exp(-(1/n) * sum_of_logs)

    def _process_text(self,text: str) -> list[list[str]]:
        '''
        Processes the text by segmenting it into sentences and tokenizing it with a BPE tokenizer
        '''
        normalized_text = normalize_text(text)
        sentences = split_sentences(normalized_text)
        tokenized_sentences = []
        for s in sentences:
            tokenized_s = self.tokenizer.tokenize(s)
            tokenized_sentences.append(tokenized_s)
        self._add_boundaries(tokenized_sentences)
        return tokenized_sentences
    
    def _add_boundaries(self, token_sentences: list[list[str]]) -> None:
        for s in token_sentences:
            for _ in range(self.N - 1):
                s.insert(0,'<s>')
            s.append('</s>')