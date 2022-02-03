# %%
import json
import random
import warnings
from pprint import pprint

import spacy
from spacy.training import *
from spacy.util import minibatch, compounding

from definitions import ROOT_DIR
from src.analysis.spacy_confusion_matrix import SpacyConfussionMatrix
from src.data.parser import DataSet

warnings.filterwarnings("ignore")
MODEL_PATH = f"{ROOT_DIR}/models"

class BaseModel:

    """
    Fine tuning an existing model.
    Prssible issues with catastrophic forgetting:
    https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
    https://stackoverflow.com/questions/65939855/spacy-custom-name-entity-recognition-ner-catastrophic-forgetting-issue

    """

    def __init__(self, initial_model = None, use_corpus="raw"):

        assert initial_model
        self._initial_model = initial_model
        self.nlp = spacy.load(self._initial_model)
        self.ds = DataSet(use_corpus=use_corpus)
        dataset = self.ds.get_train_test()
        self.train_data, self.test_data = dataset[0], dataset[1]
        self.save_path = f"{MODEL_PATH}/finetuned_{self._initial_model}_{use_corpus}"

    def train(self, epochs=30, dropout=0.5):
        """
        Fine-tune the initial_model on the provided dataset.
        Train for epochs number of times.
        Serialize best performing models, based on the f-score.
        :param epochs:
        :return:
        """

        best_fmeasure = 0

        assert self.pipe_exceptions
        unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in self.pipe_exceptions]
        optimizer = self.nlp.resume_training()

        with self.nlp.disable_pipes(*unaffected_pipes):
            print(f"Training with pipes: {self.nlp.pipe_names}")
            # Training for 30 iterations
            for iteration in range(epochs):
                print(f"##########          Training on epoch {iteration}           ##########")
                # shufling examples before every iteration
                random.shuffle(self.train_data)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(self.train_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    batch_examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc,
                                                    {"entities": annotations})
                        batch_examples.append(example)
                    self.nlp.update(
                        batch_examples,
                        drop=dropout,  # dropout - make it harder to memorise data
                        losses=losses,
                        sgd=optimizer
                    )
                print("Losses", losses)

                # store the model in case of best f-score
                # todo use CI/CD for data/model versioning (i.e. W&B etc.)
                best_fmeasure = self.evaluate(best_fmeasure)


    def evaluate(self, best_fmeasure):
        """
        Evaluate on test dataset.
        :return: scores dictionary
        """
        examples = []
        for text, annotations in self.test_data:
            doc = self.nlp.make_doc(text)
            examples.append(Example.from_dict(doc, {"entities": annotations}))

        scores = self.nlp.evaluate(examples) # This will provide overall and per entity metrics

        if scores["ents_f"] > best_fmeasure:
            best_fmeasure = scores["ents_f"]
            print(f"Saving new SOTA model with f-score of {best_fmeasure}")
            self.nlp.to_disk(self.save_path)
            with open(f"{self.save_path}/best_epoch.json", 'w') as fp:
                json.dump(scores, fp)

            cm = SpacyConfussionMatrix(self.nlp, self.test_data, self.save_path)
            cm.get_confusion_matrix()

        pprint({label: round(scores[label], 3) for label in ['ents_p', 'ents_r', 'ents_f']})
        return best_fmeasure
