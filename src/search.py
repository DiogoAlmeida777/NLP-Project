import re
import math
from nlp_utils import normalize_text, split_sentences
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class DocumentInfo:
  text: str
  length: int

@dataclass
class SearchResult:
  doc_id: int
  score: float
  text: str

class BM25Search(ABC):
  def __init__(self, k1: float = 1.2, b: float = 0.75):
    self.k1 = k1
    self.b = b
    self.inverted_index = {}
    self.documents_details: dict[int, DocumentInfo] = {}
    self.avg_dl = 0
    self.n_docs = 0

  def fit(self, corpus: str):
    sentences = split_sentences(normalize_text(corpus))
    self.n_docs = len(sentences)
    total_len = 0

    for i, sent in enumerate(sentences):
      tokens = self._tokenize(sent.lower())
      current_len = len(tokens)
      total_len += current_len

      self.documents_details[i] = DocumentInfo(text=sent, length=current_len)

      for token in tokens:
        if token not in self.inverted_index:
          self.inverted_index[token] = {}
        self.inverted_index[token][i] = self.inverted_index[token].get(i, 0) + 1
      
    self.avg_dl = total_len / self.n_docs if self.n_docs > 0 else 0
    
  @abstractmethod
  def _tokenize(self, text: str) -> list[str]:
    pass

  def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
    query_terms = self._tokenize(query.lower())
    doc_scores = {}

    for t in query_terms:
      if t not in self.inverted_index:
        continue

      df = len(self.inverted_index[t])
      idf = math.log10((self.n_docs - df + 0.5) / (df + 0.5) + 1)

      for doc_id, count in self.inverted_index[t].items():
        tf = count 
        d_len = self.documents_details[doc_id].length

        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (d_len / self.avg_dl))

        score = idf * (numerator / denominator)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
    
    results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [SearchResult(doc_id=d_id, score=s, text=self.documents_details[d_id].text) for d_id, s in results[:top_k]]
  
class BM25Words(BM25Search):
  def _tokenize(self, text: str):
    return re.findall(r'\w+', text)

class BM25BPE(BM25Search):
  def __init__(self, bpe_tokenizer, k1=1.2, b=0.75):
    super().__init__(k1, b)
    self.tokenizer = bpe_tokenizer
  
  def _tokenize(self, text: str):
    return self.tokenizer.tokenize(text)