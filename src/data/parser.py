import itertools
from typing import Tuple

import nltk
import pandas as pd
import spacy
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

from definitions import ROOT_DIR


class DataSet:

    """

    Read and split data, based on https://github.com/trent-b/iterative-stratification

In [26]: annotations_df.labels.value_counts()
Out[26]:
ORG    228
LOC    143
PER    128
PRF    111
PRD     63
EVN     25
TIM     25
DOC     23
NAT     22
DAT     20
Name: labels, dtype: int64

In [33]: df["annotation_nr"].value_counts()
Out[33]:
1     51
2     35
3     28
4     22
5     14
6     11
9     10
7      9
8      7
10     5
13     3
11     3
16     1
12     1

In [35]: df["token_nr"].value_counts()
Out[35]:
3      10
2      10
7       5
50      4
28      4
       ..
98      1
79      1
113     1
67      1
76      1
Name: token_nr, Length: 100, dtype: int64
    """

    corpora = {
        "raw": f"{ROOT_DIR}/data/data.json",
        "gold": f"{ROOT_DIR}/data/prepared_corpus/data.json"
    }

    def __init__(self, use_corpus=None):

        assert use_corpus in self.corpora

        try:
            self.data = pd.read_json(self.corpora[use_corpus])
        except ValueError:
            self.prepare_gold_corpora_nltk()
        print(f"Training on {self.corpora[use_corpus]}")
        self.data["labels"] = self.data.annotations.apply(lambda x: [a["labels"][0] for a in x])
        self.data["entities"] = self.data.annotations.apply(lambda x: [(anno["start"], anno["end"], anno['labels'][0]) for anno in x ])
        self._unique_labels = set(itertools.chain(*self.data["labels"].values.tolist()))

    def get_train_test(self, min_length=40) -> Tuple[list, list]:
        """
        Get trainng splits with text of at least min_lenght
        :return: train and test splits as lists
        """
        tmp_data  =self.data
        mask = tmp_data.data.str.len() >= min_length
        tmp_data = tmp_data.loc[mask]
        mlb = MultiLabelBinarizer()
        document_label_onehot = mlb.fit_transform(tmp_data.labels.tolist())
        mlskf = MultilabelStratifiedKFold(random_state=42, shuffle=True)
        generator_splits = mlskf.split(tmp_data.data.tolist(), document_label_onehot)
        first_split = next(generator_splits)
        train = tmp_data.iloc[first_split[0]][["data", "entities"]].values.tolist()
        test = tmp_data.iloc[first_split[1]][["data", "entities"]].values.tolist()
        return train, test

    def fix_annotation_offset(self, sentence, annotation, token_start_boundaries, token_end_mapper):
        # due to spaCy tokenization, the original annotaitons are causing an error while generating the required format for TRF models.
        # This is a hot fix and needs to be analyzed better.
        # currently issues an error

        annnotation_start = annotation['start']
        annotation_end = annotation['end']
        text = annotation['text']

        try:
            assert annnotation_start in token_start_boundaries
        except AssertionError:
            annnotation_start = [x for x in token_start_boundaries if x - annotation_end < 0][-1]
            text = " ".join([token.text for token in sentence if annnotation_start <= token.idx < annotation_end])
            print((f"{x['start']} not a token START position. Updating with {annnotation_start}"))
            print((f"Old annotation text: {x['text']}\tNew annotation text: {text}"))

        try:
            assert annotation_end in token_end_boundaries
        except AssertionError:
            annotation_end = [x for x in token_end_boundaries if x - annotation_end > 0][0]
            text = " ".join([token.text for token in sentence if annnotation_start <= token.idx < annotation_end])
            print((f"{x['end']} not a token END position. Updating with {annotation_end}"))
            print((f"Old annotation text: {x['text']}\tNew annotation text: {text}"))

        new_annotation_start = annnotation_start - sent_start
        new_annotation_end = annotation_end - sent_start
        return {
            'start': new_annotation_start,
            'end': new_annotation_end,
            'text': text,
            'labels': x['labels']
        }

    def get_sentence_annotations(self, sentence, annotations: list, sent_start: int, sent_end: int):
        """
        Filter sentece annotations and reset positional arguments.
        :param sentence:
        :param annotations:
        :return:
        """
        filtered_annotations = []
        token_start_boundaries = {token.idx: token.idx+len(token)  for token in sentence}
        token_end_boundaries = {token.idx+len(token):token.idx  for token in sentence}
        print(f"Range: {sent_start} - {sent_end}")
        for x in annotations:
            if x['start'] >= sent_start and x['end'] <= sent_end:
                x['start'] -= sentence.start_char
                x['end'] -= sentence.end_char
                filtered_annotations.append(x)
                print(f"In range: {x}")
            else:
                print(f"Not in range: {x}")
        try:
            assert filtered_annotations
        except AssertionError:
            print("bla")

        return filtered_annotations

    def prepare_gold_corpora_nltk(self):
        gold_corpus = []
        skipped = 0
        # nlp = spacy.load("de_core_news_lg")
        raw_data = pd.read_json(self.corpora['raw'])
        for index, row in raw_data.iterrows():
            text = row["data"]
            sents = nltk.sent_tokenize(text,language='german')
            for sent in sents:
                sent_start = text.find(sent)
                sent_end = sent_start + len(sent)
                sent_annotations = [x for x in row['annotations'] if x['start'] >= sent_start and x['end'] <= sent_end]
                for x in sent_annotations:
                    x['start'] -= sent_start
                    x['end'] -= sent_start
                gold_corpus.append([sent, sent_annotations])
        self.data = pd.DataFrame(gold_corpus, columns=["data", "annotations"])
        self.data.reset_index().to_json(self.corpora['gold'], orient='records', force_ascii=False)

    def prepare_gold_corpora_spacy(self):
        """
        Prepare given corpus for training.
        Split into sentences and extract sentence level annotations.
        Reset annotation positions by substracting with snt.start
        :return: set self.data to new dataframe with the same naming scheme.

        deprecated
        """
        raise ValueError("Not used currently. spaCy sentence splitting is suboptimal.")
        gold_corpus = []
        skipped = 0
        nlp = spacy.load("de_core_news_lg")
        raw_data = pd.read_json(self.corpora['raw'])
        for index, row in raw_data.iterrows():
            doc = nlp(row["data"])
            sents = doc.sents
            sentences = [i for i in doc.sents]
            if len(sentences) == 1:
                gold_corpus.append([row["data"], row['annotations']])
            else:
                for sent in sentences:
                    sent_annotations = self.get_sentence_annotations(sent, row['annotations'])
                    # if sent_annotations:
                    gold_corpus.append([sent.text, sent_annotations])
        self.data = pd.DataFrame(gold_corpus, columns=["data", "annotations"])
        self.data.reset_index().to_json(self.corpora['gold'], orient='records', force_ascii=False)

    def statistics(self):
        """
        General statistics.
        :return:
        """
        self.data["annotation_nr"] = self.data.annotations.apply(lambda x: len(x))
        self.data["token_nr"] = self.data.data.apply(lambda x: len(x.split()))

        self.data["annotation_nr"].value_counts().hist()
        self.data["token_nr"].value_counts().hist()
        self.data["labels"].value_counts().hist()


if __name__ == "__main__":
    dataset = DataSet("gold")
    dataset.prepare_gold_corpora_nltk()
