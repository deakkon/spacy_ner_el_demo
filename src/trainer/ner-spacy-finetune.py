import argparse
import warnings

from definitions import ROOT_DIR
from src.trainer import BaseModel

warnings.filterwarnings("ignore")

MODEL_PATH = f"{ROOT_DIR}/models"

class FineTunedSpacyModel(BaseModel):

    """
    Fine tuning an existing model.
    Prssible issues with catastrophic forgetting: https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
    """

    initial_model = "de_core_news_lg"

    def __init__(self, use_corpus="raw"):
        """
        Load initial_model and prepare the dataset.
        Dataset is already stratified.
        :param initial_model:
        """
        BaseModel.__init__(self, self.initial_model, use_corpus=use_corpus)
        self.pipe_exceptions = ["ner"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',
                        type=str,
                        choices=["raw", "gold"],
                        help='Train on which dataset?')
    args = parser.parse_args()
    model = FineTunedSpacyModel(args.dataset)
    model.train()
