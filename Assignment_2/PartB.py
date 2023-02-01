import spacy
from spacy.tokens import Doc
import pandas as pd
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from PartC import PartC
import csv

config = {
   "moves": None,
   "update_with_oracle_cut_size": 100,
   "learn_tokens": False,
   "min_action_freq": 30,
   "model": DEFAULT_PARSER_MODEL,
}

# def custom_tokenizer(text):
#     tokens = []
#
#     # your existing code to fill the list with tokens
#
#     # replace this line:
#     return tokens
#
#     # with this:
#     return Doc(nlp.vocab, tokens)


def custom_token(token):
    return token

class partB:
    def __init__(self):
        self.conllist_2017_trial = 'Data/conllst.2017.trial.simple.conll'

    def parse_output(self):
        df = pd.read_table(self.conllist_2017_trial, sep='\t', header=None, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True, escapechar='\\')
        df.columns = ['tokenNum', 'word', 'lemma', 'POS']
        raw_text = []
        # raw_text_list = []
        prev, curr = 0, 0
        nlp = spacy.load("en_core_web_sm")
        for index, row in df.iterrows():
            if row.tokenNum == 1 and index != 0:
                curr = index
                raw_text.append(Doc(nlp.vocab, df.word[prev:curr]))
                # raw_text_list.append(pd.Series(df.word[prev:curr]))
                prev = index
            elif index == len(df)-1:
                raw_text.append(Doc(nlp.vocab, df.word[prev:]))
                # raw_text_list.append(pd.Series(df.word[prev:curr]))
                prev = index
            #print(index, row.tokenNum)
        # nlp1 = sc.init_parser()
        # doc = Doc(nlp.vocab, "President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area .".split(" "), lemmas=df.lemma[:19].tolist(), pos=df.POS[:19].tolist())
        # doc2 = Doc(nlp.vocab, "President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area .".split(" "), lemmas=df.lemma[:19].tolist(), pos=df.POS[:19].tolist())
        # parser = nlp.get_pipe("parser")
        output = pd.DataFrame(None,None,["index","text","lemma","tag","head","dep", "dependants"])
        nlp.tokenizer = custom_token
        for sent in raw_text:
            # nlp.tokenizer = nlp.tokenizer.tokens_from_list
            # res = nlp(sent.text, disable=['tagger', 'ner', 'attribute_ruler', 'lemmatizer'])
            # toks = nlp.tokenizer(sent.text)
            res = nlp(sent)
            # for token in toks:
            #     print(f"{token.i+1}\t{token.text}\t{token.lemma_}\t{token.tag_}\t{token.head.i+1}\t{token.dep_}")
            for token in res:
                # print(f"{token.i+1}\t{token.text}\t{token.lemma_}\t{token.tag_}\t{token.head.i+1}\t{token.dep_}")
                todf = pd.DataFrame(data=[[token.i+1, token.text, token.lemma_, token.tag_, token.head.i+1, token.dep_, [str(child) for child in token.children]]]
                                    , columns=["index", "text", "lemma", "tag", "head", "dep", "dependants"])
                output = output.append(todf,ignore_index=True)
        # parser.pipe(doc)
        # scores = parser.predict([doc, doc2])
        # texts = [
        #     "Net income was $9.4 million compared to the prior year of $2.7 million.",
        #     "Revenue exceeded twelve billion dollars, with a loss of $1b.",
        # ]
        # for doc in nlp.pipe(doc, disable=["tok2vec", "tagger", "ner", "attribute_ruler", "lemmatizer"]):
        #     # Do something with the doc here
        #     print([(ent.text, ent.label_) for ent in doc.ents])
        # # This usually happens under the hood
        # processed = nlp(doc)
        # parser = nlp.add_pipe("parser", config=config)

        # # processed = parser.predict(doc)
        # for token in doc:
        #     print(token.text, token.dep_, token.head.text, token.head.pos_,
        #           [child for child in token.children])

        # print("\nBreak\n")
        # nlp = spacy.load("en_core_web_sm")
        # doc = nlp(doc)
        # for token in doc:
        #     print(token.text, token.dep_, token.head.text, token.head.pos_,
        #           [child for child in token.children])

        return output
        # return df.head()
        # return parser.predict(df.word)


if __name__ == '__main__':
    subtask_b = partB()
    df = subtask_b.parse_output()
    pd.set_option('display.max_columns', None)
    print(df.head(50))
    df.to_csv(r'Q6-Parser-Output.csv', index=False)
    partC = PartC()
    comparison_table = partC.compare_outputs(df)

    comparison_table.to_csv(r'Q8-3-Token-True-False.csv', index=False)
    print(comparison_table.head(50))
    partC.error_calculation(df)

