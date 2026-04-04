from bpe import bpe_tokenizer
from nlp_utils import *
from ngram import n_gram
from information_retrieval import bm25_search, bm25_search_words, bm25_search_bpe
from tests import test_bpe, test_n_gram, test_bm25search, test_on_generated_sentences
import sys



def main() -> None:
    with open(file="corpus/dracula_clean.txt",mode="r",encoding="utf-8") as file:
        corpus = file.read()
        file.close()

    bpe = bpe_tokenizer(corpus,k = 5000)
    bpe.learn()
    
    #with open(file="results/bpe_test.txt",mode="w") as bpe_file:
    #    original_stdout = sys.stdout
    #    try:
    #        sys.stdout = bpe_file
    #        test_bpe(bpe)
    #    finally:
    #        sys.stdout = original_stdout

    eight_gram = n_gram(8,0.4,bpe)
    eight_gram.train(corpus)

    #with open(file="results/8gram_test.txt",mode="w") as ngram_file:
    #    original_stdout = sys.stdout
    #    try:
    #        sys.stdout = ngram_file
    #        test_n_gram(eight_gram)
    #    finally:
    #        sys.stdout = original_stdout
    
    with open(file="results/8gram_test_extra.txt",mode="w") as ngram_file:
        original_stdout = sys.stdout
        try:
            sys.stdout = ngram_file
            test_on_generated_sentences(eight_gram)
        finally:
            sys.stdout = original_stdout

    #ir_words = bm25_search_words(corpus)

    #with open(file="results/bm25_words_test.txt",mode="w") as bm25words_file:
    #    original_stdout = sys.stdout
    #    try:
    #        sys.stdout = bm25words_file
    #        print("word tokenization:\n")
    #        test_bm25search(ir_words)
    #    finally:
    #        sys.stdout = original_stdout

    #ir_bpe = bm25_search_bpe(corpus,5000)

    #with open(file="results/bm25_bpe_test.txt",mode="w") as bm25bpe_file:
    #    original_stdout = sys.stdout
    #    try:
    #        sys.stdout = bm25bpe_file
    #        print("bpe tokenization:\n")
    #        test_bm25search(ir_bpe)
    #     finally:
    #        sys.stdout = original_stdout



if __name__ == "__main__":
    main()