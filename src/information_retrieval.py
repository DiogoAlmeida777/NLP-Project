import re
import math
from nlp_utils import normalize_text, split_sentences
from bpe import bpe_tokenizer
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class DocumentInfo:
  text: str
  length: int

class bm25_search(ABC):

    def __init__(self,corpus: str, k1: float = 1.2, b: float = 0.75) -> None:
        self.inverted_index: dict[str,dict[int,int]] = {}
        self.documents: dict[int,DocumentInfo] = {}
        self.n_docs: int = 0
        self.avg_dl: float = 0
        self.k1: float = k1
        self.b: float = b

        normalized_corpus = normalize_text(corpus)
        sentences = split_sentences(normalized_corpus)
        for i, s in enumerate(sentences):
            sentence_lowercase = s.lower()
            terms = self._tokenize(sentence_lowercase)
            sentence_len = len(terms)
            self.avg_dl += sentence_len
            self.documents[i] = {
                "sentence": s,
                "len": sentence_len
            }

            for t in terms:
                if t not in self.inverted_index:
                    self.inverted_index[t] = {}
                
                self.inverted_index[t][i] = self.inverted_index[t].get(i,0) + 1

        self.n_docs = len(self.documents)
        self.avg_dl /= len(sentences)
    

    @abstractmethod
    def _tokenize(self, s:str) -> list[str]:
        pass

    
    def search(self, query:str, top_k:int = 10) -> list[tuple[int,float,str]]:
        query_terms = self._tokenize(query.lower())
        doc_scores = {}

        for t in query_terms:

            if t not in self.inverted_index:
                continue
            
            df = len(self.inverted_index[t])
            idf = math.log((self.n_docs - df + 0.5)/(df + 0.5) + 1)

            for doc, count in self.inverted_index[t].items():
                f = count
                d = self.documents[doc]["len"]
                d_avg = self.avg_dl
                score = idf * (
                    (f * (self.k1 + 1)) /
                    (f + self.k1*(1-self.b + self.b*(d / d_avg)))
                )
                doc_scores[doc] = doc_scores.get(doc,0) + score
        
        results = sorted(doc_scores.items(),key=lambda x : x[1],reverse=True)

        top_k_results = []
        for doc_id, doc_score in results[:top_k]:
            sentence = self.documents[doc_id]["sentence"]
            top_k_results.append((doc_id,doc_score,sentence))
        
        return top_k_results
        

class bm25_search_bpe(bm25_search):
    def __init__(self, corpus, k_merges:int ,k1 = 1.2, b = 0.75):
        self.tokenizer = bpe_tokenizer(corpus.lower(),k_merges)
        self.tokenizer.learn()
        super().__init__(corpus, k1, b)
    
    def _tokenize(self, s):
        return self.tokenizer.tokenize(s)
    
class bm25_search_words(bm25_search):
    def _tokenize(self, s):
        return re.findall(r'\w+',s)