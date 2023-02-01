import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import random
from tqdm import tqdm


class partB:

    def __init__(self):
        self.dataset = "SemEval2018-Task3/datasets/train/SemEval2018-T3-train-taskB_emoji.txt"
        self.nlp = spacy.load("en_core_web_sm")
        self.raw = None
        self.df = pd.DataFrame(columns=["Tweet Index","Label","Tweet Text"])

    def __transform_2_df(self, raw, split_num=2):
        df = pd.DataFrame(columns=["Tweet Index","Label","Tweet Text"])
        if split_num == 2:
            for line in raw:
                prt_lin = line.split("\t", maxsplit=split_num)
                ndf = pd.DataFrame([[int(prt_lin[0]), int(prt_lin[1]), prt_lin[2].replace("\n","")],],columns=["Tweet Index","Label","Tweet Text"])
                df = df.append(ndf, ignore_index=True)
        else:
            for line in raw:
                prt_lin = line.split("\t", maxsplit=split_num)
                ndf = pd.DataFrame([[int(prt_lin[0]), prt_lin[1].replace("\n", "")], ],
                                   columns=["Tweet Index", "Tweet Text"])
                df = df.append(ndf, ignore_index=True)
        return df

    def class_dist(self,split_num = 2):
        file = open(self.dataset, encoding='utf-8')
        file.readline()
        self.raw = file.readlines()
        file.close()
        self.df = self.__transform_2_df(self.raw, split_num)
        # train_test_split(random_state=42)
        l0 = self.df[self.df["Label"] == 0].shape[0]
        l1 = self.df[self.df["Label"] == 1].shape[0]
        l2 = self.df[self.df["Label"] == 2].shape[0]
        l3 = self.df[self.df["Label"] == 3].shape[0]

        print(f"Class label: 0, Instances: {l0}, Frequency: {round((l0/3834)*100, 2)}%, Example: {self.df[self.df['Label'] == 0].iloc[0,2]}")
        print(f"Class label: 1, Instances: {l1}, Frequency: {round((l1/3834)*100, 2)}%, Example: {self.df[self.df['Label'] == 1].iloc[0,2]}")
        print(f"Class label: 2, Instances: {l2}, Frequency: {round((l2/3834)*100, 2)}%, Example: {self.df[self.df['Label'] == 2].iloc[0,2]}")
        print(f"Class label: 3, Instances: {l3}, Frequency: {round((l3/3834)*100, 2)}%, Example: {self.df[self.df['Label'] == 3].iloc[0,2]}")

    def baselines(self, split_num=2, majority = False):
        choices = [0, 1, 2, 3]

        file = open("SemEval2018-Task3/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt", encoding='utf-8')
        file.readline()
        raw_gold = file.readlines()
        file.close()
        refined_gold = self.__transform_2_df(raw_gold, split_num)

        file = open("SemEval2018-Task3/datasets/test_TaskB/SemEval2018-T3_input_test_taskB_emoji.txt", encoding='utf-8')
        file.readline()
        raw_silver = file.readlines()
        file.close()
        refined_silver = self.__transform_2_df(raw_silver, 1)

        random.seed(42)

        acc_0, acc_1, acc_2, acc_3 = 0, 0, 0, 0
        acc_mic, prec_mic, rec_mic, f1_mic = 0, 0, 0, 0
        acc_weight, prec_weight, rec_weight, f1_weight = 0, 0, 0, 0
        prec, rec, f1 = 0, 0, 0
        for i in tqdm(range(100)):
            for j in range(len(refined_silver.iloc[:,1])):
                if majority:
                    refined_silver.iloc[:, 1][j] = 0
                else:
                    refined_silver.iloc[:, 1][j] = random.choice(choices)
            guesses_0 = np.nan_to_num(refined_silver.iloc[:, 1].astype(int).where(refined_silver.iloc[:, 1] == 0).to_numpy(), nan=-1)
            truth_0 = np.nan_to_num(refined_gold.iloc[:, 1].astype(int).where(refined_gold.iloc[:, 1] == 0).to_numpy(), nan=-1)
            guesses_1 = np.nan_to_num(refined_silver.iloc[:, 1].astype(int).where(refined_silver.iloc[:, 1] == 1).to_numpy(), nan=-1)
            truth_1 = np.nan_to_num(refined_gold.iloc[:, 1].astype(int).where(refined_gold.iloc[:, 1] == 1).to_numpy(), nan=-1)
            guesses_2 = np.nan_to_num(refined_silver.iloc[:, 1].astype(int).where(refined_silver.iloc[:, 1] == 2).to_numpy(), nan=-1)
            truth_2 = np.nan_to_num(refined_gold.iloc[:, 1].astype(int).where(refined_gold.iloc[:, 1] == 2).to_numpy(), nan=-1)
            guesses_3 = np.nan_to_num(refined_silver.iloc[:, 1].astype(int).where(refined_silver.iloc[:, 1] == 3).to_numpy(), nan=-1)
            truth_3 = np.nan_to_num(refined_gold.iloc[:, 1].astype(int).where(refined_gold.iloc[:, 1] == 3).to_numpy(), nan=-1)


            guess = refined_silver.iloc[:, 1].astype(int).to_numpy()
            truth = refined_gold.iloc[:, 1].astype(int).to_numpy()

            acc_0 += accuracy_score(truth_0, guesses_0)
            acc_1 += accuracy_score(truth_1, guesses_1)
            acc_2 += accuracy_score(truth_2, guesses_2)
            acc_3 += accuracy_score(truth_3, guesses_3)
            acc_mic += accuracy_score(truth, guess)
            acc_weight += balanced_accuracy_score(truth, guess)

            prec += precision_score(truth, guess, average=None, zero_division=0)
            prec_mic += precision_score(truth, guess, labels=[0, 1, 2, 3], average="macro", zero_division=0)
            prec_weight += precision_score(truth, guess, labels=[0, 1, 2, 3], average="weighted", zero_division=0)
            rec += recall_score(truth, guess, average=None, zero_division=0)
            rec_mic += recall_score(truth, guess, labels=[0, 1, 2, 3], average="macro", zero_division=0)
            rec_weight += recall_score(truth, guess, labels=[0, 1, 2, 3], average="weighted", zero_division=0)
            f1 += f1_score(truth, guess, average=None, zero_division=0)
            f1_mic += f1_score(truth, guess, labels=[0, 1, 2, 3], average="macro", zero_division=0)
            f1_weight += f1_score(truth, guess, labels=[0, 1, 2, 3], average="weighted", zero_division=0)
            # test = classification_report(truth, guess, labels=[0, 1, 2, 3]) #this could be important
            # print(test)

        if majority:
            print("\nFor majority:\n")
        else:
            print("\nFor random:\n")
        print(f"For label 0: Acc = {round(acc_0 / 100, 2)}, Prec = {round(prec[0] / 100, 2)}, Rec = {round(rec[0] / 100, 2)}, F1 = {round(f1[0] / 100, 2)}")
        print(f"For label 1: Acc = {round(acc_1 / 100, 2)}, Prec = {round(prec[1] / 100, 2)}, Rec = {round(rec[1] / 100, 2)}, F1 = {round(f1[1] / 100, 2)}")
        print(f"For label 2: Acc = {round(acc_2 / 100, 2)}, Prec = {round(prec[2] / 100, 2)}, Rec = {round(rec[2] / 100, 2)}, F1 = {round(f1[2] / 100, 2)}")
        print(f"For label 3: Acc = {round(acc_3 / 100, 2)}, Prec = {round(prec[3] / 100, 2)}, Rec = {round(rec[3] / 100, 2)}, F1 = {round(f1[3] / 100, 2)}")
        print(f"For macro: Acc = {round(acc_mic/ 100, 2)}, Prec = {round(prec_mic / 100, 2)}, Rec = {round(rec_mic / 100, 2)}, F1 = {round(f1_mic / 100, 2)}")
        print(f"For weighted: Acc = {round(acc_weight / 100, 2)}, Prec = {round(prec_weight / 100, 2)}, Rec = {round(rec_weight / 100, 2)}, F1 = {round(f1_weight / 100, 2)}")


if __name__ == '__main__':
    subtaskB = partB()
    subtaskB.class_dist()
    subtaskB.baselines()
    subtaskB.baselines(majority=True)