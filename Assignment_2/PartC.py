import spacy
import pandas as pd
from spacy.tokens import Doc
import csv


class PartC:
    def __init__(self):
        self.conllist_2017_trial = 'Data/conllst.2017.trial.simple.dep.conll'

    def get_dependants(self, index, sentence_start,  df):
        dependents = []
        head = str(df.iloc[index][0])
        max_ind = 0
        while sentence_start < len(df) and df.iloc[sentence_start][0] > max_ind:
            max_ind = df.iloc[sentence_start][0]
            if str(df.iloc[sentence_start][4]) == head and sentence_start != index:
                dependents.append(df.iloc[sentence_start][1])
            sentence_start += 1

        return dependents

    def compare_outputs(self, b_output):
        df = pd.read_table(self.conllist_2017_trial, sep='\t', header=None, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True, escapechar='\\')
        df.columns = ["index", "text", "lemma", "tag", "head", "dep"]
        found_all_tokens = []
        found_all_roots = []
        raw_text = []
        output_table = pd.DataFrame(None, None, ["output", "gold", "same.head", "same.label", "same.dependents"])
        curr, prev = 0, 0
        nlp = spacy.load("en_core_web_sm")
        for df_index, row in df.iterrows():
            dependents = self.get_dependants(df_index, prev, df)
            todf = pd.DataFrame(data=[[b_output.iloc[df_index][1], row[1], str(b_output.iloc[df_index][4]) == str(row[4]),
                                       str(b_output.iloc[df_index][5]).upper() == str(row[5]).upper(),
                                       dependents == b_output.iloc[df_index][6]]]
                                , columns=["output", "gold", "same.head", "same.label", "same.dependents"])
            output_table = output_table.append(todf, ignore_index=True)
            if row[0] == 1 and df_index != 0:
                curr = df_index
                df_tok = row[0]
                out_tok = b_output.iloc[df_index][0]
                raw_text.append(Doc(nlp.vocab, df.text[prev:curr]))
                if df_tok != out_tok:
                    found_all_tokens.append(len(raw_text))
                depen = b_output[prev:curr].dep
                if "ROOT" not in depen.unique():
                    found_all_roots.append(len(raw_text))
                # raw_text_list.append(pd.Series(df.word[prev:curr]))
                prev = df_index
            elif df_index == len(df) - 1:
                raw_text.append(Doc(nlp.vocab, df.text[prev:]))
                df_tok = row[0]
                out_tok = b_output.iloc[df_index][0]
                if df_tok != out_tok:
                    found_all_tokens.append(len(raw_text))
                depen = b_output[prev:].dep
                if "ROOT" not in depen.unique():
                    found_all_roots.append(len(raw_text))
                # raw_text_list.append(pd.Series(df.word[prev:curr]))
                prev = df_index
        if len(found_all_tokens) > 0:
            print("Found token count errors in sentences: ", found_all_tokens)
        else:
            print("Found no errors in token count.")
        if len(found_all_roots) > 0:
            print("Found missing root in sentences: ", found_all_tokens)
        else:
            print("Found root in every sentence.")
        # print(output_table)
        return output_table

    def error_calculation(self, b_output):
        df = pd.read_table(self.conllist_2017_trial, sep='\t', header=None, quoting=csv.QUOTE_NONE, quotechar='"', doublequote=True, escapechar='\\')
        df.columns = ["index", "text", "lemma", "tag", "head", "dep"]
        POS_tag_list = {}
        POS_tag_error_list = {}
        POS_tag_error_percentage = {}
        LABEL_list = {}
        LABEL_error_list = {}
        LABEL_error_percentage = {}
        print("POS tag: Error rate")
        for df_index, row in df.iterrows():
            POS_tag_list[row.tag] = 0
            POS_tag_error_list[row.tag] = 0
            POS_tag_error_percentage[row.tag] = 0
            LABEL_list[row.dep] = 0
            LABEL_error_list[row.dep] = 0
            LABEL_error_percentage[row.dep] = 0
        for tag in POS_tag_list:
            x = df.loc[df['tag'] == tag]
            for select_index, row in df.loc[df['tag'] == tag].iterrows():
                POS_tag_list[str(tag)] += 1
                if b_output.iloc[select_index, 3] != str(tag):
                    POS_tag_error_list[str(tag)] += 1
            # print(f"{tag}\t{round((POS_tag_error_list.get(str(tag))/POS_tag_list.get(str(tag)))*100, 2)}%")
            POS_tag_error_percentage[str(tag)] = round((POS_tag_error_list.get(str(tag))/POS_tag_list.get(str(tag)))*100, 2)
        ordered_POS = dict(sorted(POS_tag_error_percentage.items(), key=lambda item: item[1], reverse=True))
        print(ordered_POS)
        for dep in LABEL_list:
            x = df.loc[df['dep'] == dep]
            for select_index, row in df.loc[df['dep'] == dep].iterrows():
                LABEL_list[str(dep)] += 1
                if b_output.iloc[select_index, 5].upper() != str(dep).upper():
                    LABEL_error_list[str(dep)] += 1
            # print(f"{tag}\t{round((POS_tag_error_list.get(str(tag))/POS_tag_list.get(str(tag)))*100, 2)}%")
            LABEL_error_percentage[str(dep)] = round((LABEL_error_list.get(str(dep))/LABEL_list.get(str(dep)))*100, 2)
        print("LABEL: Error rate")
        ordered_LABEL = dict(sorted(LABEL_error_percentage.items(), key=lambda item: item[1], reverse=True))
        print(ordered_LABEL)
        with open('POS_ERROR_ORDERED.txt', 'w') as f:
            print(ordered_POS, file=f)
        with open('LABEL_ERROR_ORDERED.txt', 'w') as f:
            print(ordered_LABEL, file=f)


