import traceback
import warnings

from spacy.tokens import DocBin
from spacy.training import Example
from tqdm import tqdm

from definitions import ROOT_DIR
from src.data.parser import DataSet
from src.trainer import BaseModel

warnings.filterwarnings("ignore")

MODEL_PATH = f"{ROOT_DIR}/models"

class FineTunedSpacyTRFModel(BaseModel):

    """
    Fine tuning an existing model.
    Prssible issues with catastrophic forgetting: https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
    """

    initial_model = "de_dep_news_trf"

    def __init__(self):
        """
        Load initial_model and prepare the dataset.
        Dataset is already stratified.
        :param initial_model:
        """
        BaseModel.__init__(self, self.initial_model, use_corpus="gold")
        # ner = self.nlp.add_pipe("ner", last=True)
        # ner.initialize(self._initilize_ner)
        # self.pipe_exceptions = ["ner"]

        # NOT LEARNING ANYTHING

    def _initilize_ner(self):
        return iter(Example.from_dict(self.nlp.make_doc(text), {"entities": annotations}) for text, annotations in self.train_data)

    def create_doc_bin(self, data: list, name: str = None):
        db = DocBin()
        issues = 0
        for text, annotations in tqdm(data, desc="Creating binary file for CLI Spacy training.", total=len(data)):
            try:
                doc = self.nlp(text)
                ents = []
                for start, end, label in annotations:
                    span = doc.char_span(start, end, label=label)
                    ents.append(span)
                try: # toknization has some issues
                    doc.ents = ents
                    db.add(doc)
                except ValueError:
                    issues += 1
                    print(issues)
            except TypeError:
                print(f"{text}\n {annotations}\n=====")

        db.to_disk(f"{ROOT_DIR}/data/docbin/{name}.spacy")

if __name__ == "__main__":
    model = FineTunedSpacyTRFModel()
    # model.train()
    model.create_doc_bin(model.train_data, "train")
    model.create_doc_bin(model.test_data, "test")
