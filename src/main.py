from bpe import bpe_tokenizer
from nlp_utils import *
from ngram import n_gram
from information_retrieval import bm25_search, bm25_search_words, bm25_search_bpe



def test_bpe(model: bpe_tokenizer):

    test_sentences = [
        "I did not sleep well, though my bed was comfortable enough, for I had all sorts of queer dreams.",
        "All day long we seemed to dawdle through a country which was full of beauty of every kind.",
        "They are, however, I am told, very harmless and rather wanting in natural self-assertion.",
        "It was on the dark side of twilight when we got to Bistritz, which is a very interesting old place.",
        "Being practically on the frontier--for the Borgo Pass leads from it into Bukovina--it has had a very stormy existence, and it certainly shows marks of it."
        "Fifty years ago a series of great fires took place, which made terrible havoc on five separate occasions.",
        "At the very beginning of the seventeenth century it underwent a siege of three weeks and lost 13,000 people, the casualties of war proper being assisted by famine and disease.",
        "Count Dracula had directed me to go to the Golden Krone Hotel, which I found, to my great delight, to be thoroughly old-fashioned, for of course I wanted to see all I could of the ways of the country.",
        "When I got on the coach the driver had not taken his seat, and I saw him talking with the landlady.",
        "A key was turned with the loud grating noise of long disuse, and the great door swung back."
    ]

    for s in test_sentences:
        print("original sentence:",s,sep="\n\n",end="\n\n")
        normalized_sentence = normalize_text(s)
        word_tokens = tokenize_sentences([normalized_sentence])[0]
        print("word tokens:",word_tokens,sep="\n\n",end="\n\n")
        print("bpe representation:",model.tokenize(s),sep="\n\n",end="\n\n")


def test_n_gram(model: n_gram):
    print("Vocabulary Size:",len(model.vocab),sep=" ",end="\n\n")
    print("Number of bigrams:",len(model.ngram_counts[1]),sep=" ",end="\n\n")
    print("Number of trigrams:",len(model.ngram_counts[2]),sep=" ",end="\n\n")
    print("Number of fourgrams:",len(model.ngram_counts[3]),sep=" ",end="\n\n")
    print("Number of fivegrams:",len(model.ngram_counts[4]),sep=" ",end="\n\n")
    print("Number of sixgrams:",len(model.ngram_counts[5]),sep=" ",end="\n\n")
    print("Number of seven_grams:",len(model.ngram_counts[6]),sep=" ",end="\n\n")
    print("Number of eightgrams:",len(model.ngram_counts[7]),sep=" ",end="\n\n")
    
    test_sentences = [
        "Count Dracula carriage arrived at Bistritz.",
        "I found the Castle key in the dark room.",
        "Vampires scream very loud when exposed under the sun"
    ]

    for s in test_sentences:
        print("sentence:",s,sep=" ",end="\n")
        print("sentence score:",model.sentence_score(s),sep=" ",end="\n\n")
    
    temperatures = [0.1,0.4,0.7,0.9,1.0,1.1,1.3,1.6,1.9,2.0]

    for t in temperatures:
        print("generated sentence with temperature =:",t,sep=" ",end="\n\n")
        print(model.generate_sentence(30,t),end="\n\n")
    
    test_input1 = """The Castle.--The grey of the morning has passed, and the sun is
                    high over the distant horizon, which seems jagged, whether with trees or
                    hills I know not, for it is so far off that big things and little are
                    mixed. I am not sleepy, and, as I am not to be called till I awake,
                    naturally I write till sleep comes. There are many odd things to put
                    down, and, lest who reads them may fancy that I dined too well before I
                    left Bistritz, let me put down my dinner exactly. I dined on what they
                    called “robber steak”--bits of bacon, onion, and beef, seasoned with red
                    pepper, and strung on sticks and roasted over the fire, in the simple
                    style of the London cat’s meat! The wine was Golden Mediasch, which
                    produces a queer sting on the tongue, which is, however, not
                    disagreeable. I had only a couple of glasses of this, and nothing else."""
    
    test_input2 = """Once upon a time there was a lovely princess. But she had an enchantment upon her of a fearful
                    sort, which could only be broken by Love's first kiss. She was locked away in a castle guarded by a
                    terrible fire breathing dragon. Many brave knights had attempted to free her from this dreadful
                    prison, but none prevailed. She waited in the dragon's keep in the highest room of the tallest tower
                    for her true love and true love's first kiss. Like that's ever going to happen. What a loony. Shrek
                    Beware Stay out I think he's in here. All right. Lets get it! Hold on. Do you know what that thing can
                    do to you? Yeah. He'll groan into your bones for his brains. Well actually that would be a giant. Now
                    Ogres, huh, they are much worse. They'll make a soup from your freshly peeled skin."""

    print("excerpt from the corpus (Dracula):",end="\n\n")
    print("perplexity = ",model.perplexity(test_input1),sep=" ",end="\n\n")

    print("excerpt from Shrek's script:",end="\n\n")
    print("perplexity = ",model.perplexity(test_input2),sep=" ",end="\n\n")


def test_bm25search(model:bm25_search):
    character_queries = [
        "Dracula",
        "Jonathan Harker",
        "Van Helsing",
        "Lucy",
        "Agatha",
        "Seward",
        "Mina"
    ]

    rare_words_queries = [
        "quondam",
        "prodigal",
        "queer",
        "miasma",
        "diabolical",
        "crucifix",
        "berserker",
        "lugubrious",
        "boyar",
        "calèche"
    ]

    morphologically_related_forms = [
        "conscious",
        "unconscious",
        "attended",
        "unattended",
        "happy",
        "unhappy",
        "run",
        "running",
        "open",
        "opened",
        "unopened",
        "easy",
        "uneasy",
        "easily",
        "uneasily",
        "sun",
        "sunny",
        "vampire",
        "vampirism",
        "bite",
        "biting",
        "bitten"
    ]

    short_keyword_queries = [
        "dracula castle",
        "dark room",
        "sucking blood",
        "vampire bite",
        "wooden stake",
        "curse of immortality",
        "Bistritz",
        "Jonathan Harker diary"
    ]

    full_sentence_queries = [
        "sunlight, crucifixes and garlic are weaknesses of vampires.",
        "vampires bite and suck the blood of their victims.",
        "Count Dracula sleeps inside a coffin during daylight.",
        "what causes lucy illness?",
        "It was on the dark side of twilight when we got to Bistritz, which is a very interesting old place.",
        "Dracula moves like a lizard.",
        "Mina Harker transcribes the journals with a typewriter."
    ]

    query_types = [
        character_queries,
        rare_words_queries,
        morphologically_related_forms,
        short_keyword_queries,
        full_sentence_queries
    ]

    query_names = [
        "Character Names", 
        "Rare Words",
        "Morphologically Related Forms",
        "Short Keywords",
        "Full Sentences"
    ]

    for queries, q_type in zip(query_types, query_names):
        print(q_type,end="\n\n")
        for q in queries:
            print("query: ",q,end="\n\n")
            top_results = model.search(q)
            for result in top_results:
                print(result,end="\n\n")
        

def main() -> None:
    with open(file="corpus/dracula_clean.txt",mode="r",encoding="utf-8") as file:
        corpus = file.read()
        file.close()
    bpe = bpe_tokenizer(corpus,k = 5000)
    bpe.learn()
    test_bpe(bpe)
    eight_gram = n_gram(8,0.4,bpe)
    eight_gram.train(corpus)
    test_n_gram(eight_gram)
    ir_words = bm25_search_words(corpus)
    print("word tokenization:\n")
    test_bm25search(ir_words)
    ir_bpe = bm25_search_bpe(corpus,5000)
    print("bpe tokenization:\n")
    test_bm25search(ir_bpe)


if __name__ == "__main__":
    main()