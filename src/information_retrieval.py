import re
import math
from nlp_utils import normalize_text, split_sentences

class bm25_search:
    
    def __init__(self,corpus: str, mode: str = 'words'):
        self.inverted_index_words: dict[str,dict[int,int]] = {}
        self.documents: dict[int,dict[str,str|int]] = {}
        self.n_docs: int = 0
        self.avg_sentence_len: float = 0
        self.k1: float = 1.2
        self.b: float = 0.75

        if mode == 'bpe':
            pass

        normalized_corpus = normalize_text(corpus)
        sentences = split_sentences(normalized_corpus)
        for i, s in enumerate(sentences):
            sentence_lowercase = s.lower()
            words = re.findall(r'\w+',sentence_lowercase)
            sentence_len = len(words)
            self.avg_sentence_len += sentence_len
            self.documents[i] = {
                "sentence": s,
                "len": sentence_len
            }

            for w in words:
                if w not in self.inverted_index_words:
                    self.inverted_index_words[w] = {}
                
                self.inverted_index_words[w][i] = self.inverted_index_words[w].get(i,0) + 1

        self.n_docs = len(self.documents)
        self.avg_sentence_len /= len(sentences)


    def search_words(self, query:str, top_k:int = 10) -> list[tuple[int,float,str]]:
        lowercased_query = query.lower()
        query_terms = re.findall(r'\w+',lowercased_query)
        doc_scores = {}

        for t in query_terms:

            if t not in self.inverted_index_words:
                continue
            
            df = len(self.inverted_index_words[t])
            idf = math.log10(self.n_docs/df)

            for doc, count in self.inverted_index_words[t].items():
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
        


    def search_bpe(self, query:str, top_k:int = 10) -> list[tuple[int,float,str]]:
        pass

