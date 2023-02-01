import re


def process_data(regx,text, removeNewLine:bool = True) -> str:
    processed = ""
    for i in text:
        nl = re.sub(regx, "", i)
        if removeNewLine:
            nl = nl.replace("\n", " ")
        processed += nl
    return processed


def get_types(text) -> set:
    types = set()
    for i in text:
        types.add(str.upper(i.text))
    return types


def get_words(text) -> list:
    words = []
    for i in text:
        if not i.is_punct:
            words.append(i)
    return words
