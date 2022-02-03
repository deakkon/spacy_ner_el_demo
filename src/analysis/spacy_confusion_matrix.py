import srsly
import typer
import warnings
from pathlib import Path
import spacy
import numpy
import os
import pandas as pd

from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from spacy.training import offsets_to_biluo_tags


class SpacyConfussionMatrix:

    # taken from https://github.com/explosion/spaCy/discussions/9055

    def __init__(self, nlp, samples, output_dir):
        self.nlp = nlp
        self.samples = samples
        self.output_dir = output_dir

    def _get_cleaned_label(self, label: str):
        if "-" in label:
            return label.split("-")[1]
        else:
            return label


    def _create_total_target_vector(self):
        target_vector = []
        for sample in self.samples:
            doc = self.nlp.make_doc(sample[0])
            ents = sample[1]
            bilou_ents = offsets_to_biluo_tags(doc, ents)
            vec = [self._get_cleaned_label(label) for label in bilou_ents]
            target_vector.extend(vec)
        return target_vector

    def _get_all_ner_predictions(self, text):
        doc = self.nlp(text)
        entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
        bilou_entities = offsets_to_biluo_tags(doc, entities)
        return bilou_entities

    def _create_prediction_vector(self, text):
        return [self._get_cleaned_label(prediction) for prediction in self._get_all_ner_predictions(text)]

    def _create_total_prediction_vector(self):
        prediction_vector = []
        for i in range(len(self.samples)):
            sample = self.samples[i]
            prediction_vector.extend(self._create_prediction_vector(sample[0]))
        return prediction_vector

    def _plot_confusion_matrix(self, cm, classes, normalize=False, text=True, cmap=pyplot.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        title = "Confusion Matrix"

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

        fig, ax = pyplot.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=numpy.arange(cm.shape[1]),
               yticks=numpy.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        if text:
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax, pyplot

    def get_confusion_matrix(self):
        classes = sorted(set(self._create_total_target_vector()))
        y_true = self._create_total_target_vector()
        y_pred = self._create_total_prediction_vector()
        matrix = confusion_matrix(y_true, y_pred, labels=classes)
        # print("Generated confusion matrix!")
        cm_df = pd.DataFrame(matrix, columns=classes)
        cm_df.insert(0, "TARGETS", classes)
        ax, plot = self._plot_confusion_matrix(matrix, classes, normalize=True, text=False)
        # print("Plotted confusion matrix!")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # print(f"Saving confusion matrix data to: {self.output_dir}/confusion.csv")
        cm_df.to_csv(f"{self.output_dir}/confusion.csv")

        # print(f"Saving rendered image to: {self.output_dir}/confusion.png")
        pyplot.savefig(f"{self.output_dir}/confusion.png")
