import re


def normalize_text(text: str) -> str:
    '''
    Subtitutes multiple whitespaces into a single whitespace.
    '''
    result = re.sub(r'\s+',' ',text)
    return result

def split_sentences(text: str) -> list[str]:
    '''
    Splits a text into a list of sentences, ignoring empty sentences.
    '''
    # (?<=[]) positive lookbehind:
    # quando encontra o \s só aceita o match se o caractere anterior da match à condição
    # nao consome o caractere da condição
    segments = re.split(r'(?<=[.!?])\s',text)
    # apenas aceita segmentos não vazios.
    result = [s for s in segments if re.search(r'\w',s)]
    return result

def tokenize_sentences(sentences: list[str]) -> list[list[str]]:
    '''
    Receives a list of sentences and splits them into word and punctuation tokens
    '''
    result = []
    for s in sentences:
        result.append(re.findall(r'\w+(?:’\w+)?|[.,!?;:()\"\'’-]',s))
    return result
