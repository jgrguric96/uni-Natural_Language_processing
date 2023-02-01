import spacy, re
from pathlib import Path


class POS:

    def __init__(self, finegrain, universal, occurrences=0, tokens=None):
        self.finegrain = finegrain
        self.universal = universal
        self.occurrences = occurrences
        self.tokens = tokens
        if tokens is None:
            self.tokens = dict()

    def add_token(self, token):
        if self.tokens.get(token):
            self.tokens[token] += 1
        else:
            self.tokens[token] = 1


class partA:

    def __init__(self):
        self.dataset = "SemEval2018-Task3/datasets/train/SemEval2018-T3-train-taskB.txt"
        # python -m spacy download en_core_web_sm - provjeri ako mogu to forsirati u kodu
        # Nije potrebno na kraju to provjeriti. Zadatak zahtjeva jednostavne Sub_*.py fajlove sa izoliranim kodom
        self.nlp = spacy.load("en_core_web_sm")
        self.raw = None

    def __process_data(self, regx, text, removeNewLine: bool = True) -> str:
        processed = ""
        for i in text:
            nl = re.sub(regx, "", i)
            if removeNewLine:
                nl = nl.replace("\n", " ")
            processed += nl
        return processed

    def __get_types(self, text) -> set:
        types = set()
        for i in text:
            types.add(str.upper(i.text))
        return types

    def __get_words(self, text) -> list:
        words = []
        for i in text:
            if not i.is_punct:
                words.append(i)
        return words

    def tokenization(self):
        file = open(Path(self.dataset), errors='ignore')
        file.readline()
        raw = file.readlines()
        self.raw = raw
        file.close()

        text = self.__process_data(r'([\d]+	){2}', raw, removeNewLine=True)

        doc = self.nlp(text)
        print(f"Number of tokens: {len(doc)}")

        types = self.__get_types(doc)
        print(f"Number of types: {len(types)}")

        words = self.__get_words(doc)
        print(f"Number of words: {len(words)}")

        print(f"Average number of words per tweet: {round(len(words)/len(raw), 2)}")

        wordlen = 0
        for word in words:
            wordlen += len(word)
        print(f"Average word length: {round(wordlen/len(words), 2)}\n")

    def pos_tagging(self):
        postags = dict()
        doc = self.nlp(self.__process_data(r'([\d]+	){2}', self.raw, removeNewLine=True))
        for token in doc:
            if postags.get(token.tag_):
                postags[token.tag_].occurrences += 1
                postags[token.tag_].add_token(token.text)
            else:
                postags[token.tag_] = POS(finegrain=token.tag_, universal=token.pos_, occurrences=1, tokens={token.text:1})
        print("Finegrained POS-tag\tUniversal POS-tag\tOccurrences\tRelative Tag Frequency\t3 most frequent tokens\tExample infrequent token")
        listed = dict()
        for i in range(0, 10):
            max_occ = 0
            max_pos = ""
            for val in postags:
                if postags.get(val).occurrences >= max_occ and postags.get(val).finegrain not in listed:
                    max_occ = postags.get(val).occurrences
                    max_pos = postags.get(val).finegrain
            listed[max_pos] = max_occ
        for finegrain in listed:
            val = postags.get(finegrain)
            rnkdict = dict(sorted(val.tokens.items(), key=lambda item: item[1], reverse=True))
            rnk = list(rnkdict.keys())
            print(f"{postags.get(finegrain).finegrain}\t\t\t{postags.get(finegrain).universal}\t\t\t{postags.get(finegrain).occurrences}\t\t\t{round(postags.get(finegrain).occurrences/len(doc)*100,2)}%\t\t\t{rnk[:3]}\t\t\t{rnk[-1]}")

    def lemmatization(self):
        #{lema:{form:sentence}}
        tweets = list(self.__process_data(r'([\d]+	){2}', self.raw, removeNewLine=False).split("\n"))
        lemmas = dict()
        example = dict()
        is_satisfied = False
        for tweet in tweets:
            doc = self.nlp(tweet.replace("\n", ""))
            for token in doc:
                if lemmas.get(token.lemma_):
                    nlem = lemmas.get(token.lemma_)
                    nlem[token.text] = tweet
                    lemmas[token.lemma_] = nlem
                    if len(lemmas.get(token.lemma_)) > 2:
                        example = {token.lemma_: lemmas.get(token.lemma_)}
                        is_satisfied = True
                        break
                else:
                    lemmas[token.lemma_] = {token.text:tweet}
            if is_satisfied:
                break
        print("\nAn example for a lemma that occurs in more than two inflections in the data set:")
        print(f"Lemma: {list(example.keys())[0]}")
        print(f"Inflected forms: {list(list(example.values())[0].keys())}")
        print(f"Example sentences: {list(list(example.values())[0].values())}")

    def entity_recognition(self):
        tweets = list(self.__process_data(r'([\d]+	){2}', self.raw, removeNewLine=False).split("\n"))
        unique_entities = set()
        total_entities = []
        for tweet in tweets[:3]:
            doc = self.nlp(tweet.replace("\n", ""))
            print(f"\nTweet: {tweet}")
            print("Entities in tweet:")
            for ent in doc.ents:
                total_entities.append([ent.text, ent.label_])
                unique_entities.add(ent.label_)
                print(ent.text, ent.label_)
        print(f"\nNumber of named entities: {len(total_entities)}")
        print(f"Named entities: {total_entities}\n")
        print(f"Number of different entity labels: {len(unique_entities)}")
        print(f"Unique entity labels: {unique_entities}")


if __name__ == '__main__':
    subtaskA = partA()
    subtaskA.tokenization()
    subtaskA.pos_tagging()
    subtaskA.lemmatization()
    subtaskA.entity_recognition()

