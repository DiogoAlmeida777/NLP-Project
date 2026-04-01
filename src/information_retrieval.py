import re
import math
from nlp_utils import normalize_text, split_sentences
from bpe import bpe_tokenizer
from abc import ABC, abstractmethod

class bm25_search(ABC):

    def __init__(self,corpus: str):
        self.inverted_index: dict[str,dict[int,int]] = {}
        self.documents: dict[int,dict[str,str|int]] = {}
        self.n_docs: int = 0
        self.avg_sentence_len: float = 0
        self.k1: float = 1.2
        self.b: float = 0.75


        normalized_corpus = normalize_text(corpus)
        sentences = split_sentences(normalized_corpus)
        for i, s in enumerate(sentences):
            sentence_lowercase = s.lower()
            terms = self._tokenize(sentence_lowercase)
            sentence_len = len(terms)
            self.avg_sentence_len += sentence_len
            self.documents[i] = {
                "sentence": s,
                "len": sentence_len
            }

            for t in terms:
                if t not in self.inverted_index:
                    self.inverted_index[t] = {}
                
                self.inverted_index[t][i] = self.inverted_index[t].get(i,0) + 1

        self.n_docs = len(self.documents)
        self.avg_sentence_len /= len(sentences)
    

    @abstractmethod
    def _tokenize(self, s:str) -> list[str]:
        pass

    
    def search(self, query:str, top_k:int = 10) -> list[tuple[int,float,str]]:
        lowercased_query = query.lower()
        query_terms = self._tokenize(lowercased_query)
        doc_scores = {}

        for t in query_terms:

            if t not in self.inverted_index:
                continue
            
            df = len(self.inverted_index[t])
            idf = math.log10(self.n_docs/df)

            for doc, count in self.inverted_index[t].items():
                tf = 1 + math.log10(count)
                d = self.documents[doc]["len"]
                d_avg = self.avg_sentence_len
                score = idf * (
                    (tf * (self.k1 + 1)) /
                    (tf + self.k1*(1-self.b + self.b*(d / d_avg)))
                )
                doc_scores[doc] = doc_scores.get(doc,0) + score
        
        results = sorted(doc_scores.items(),key=lambda x : x[1],reverse=True)

        top_k_results = []
        for doc_id, doc_score in results[:top_k]:
            sentence = self.documents[doc_id]["sentence"]
            top_k_results.append((doc_id,doc_score,sentence))
        
        return top_k_results
        

class bm25_search_bpe(bm25_search):
    def __init__(self, corpus, k: int):
        self.tokenizer = bpe_tokenizer(corpus.lower(),k)
        self.tokenizer.learn()
        super().__init__(corpus)
    
    def _tokenize(self, s):
        return self.tokenizer.tokenize(s)
    
class bm25_search_words(bm25_search):
    def _tokenize(self, s):
        return re.findall(r'\w+',s)