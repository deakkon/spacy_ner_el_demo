# Introduction

MVP solution, fosucing on NLP for IE. Established pipeline made of of NER and NEN/EL components. 
EL supports SPARQL DBpedia enpoint for one concept type (Person). 
Solution is based on the spaCy universe components.

# Installation

## Create a virutal environment:

### Anaconda (prefered)

1. _conda env create --file environment.yml_
2. _conda activate lakatu_
   
### pip
   
Alternatevily, use your virtenv tool of choice (e.g. _virtualenv_) to 1) create and 2) activate your own virtenv. Then install all dependencies with

_pip install -r environment.txt_

## Install required spaCy models

Assuming you have activated the created virt env, and you are in the root of the project, run _./install_spacy.sh_ to install the required (spacy) models. This will download two pretrained models:

1. _de_core_news_lg_ used in baseline and fine-tuned model;
2. _bert-base-german-cased_ used in the custom TRF pipeline.

# Running training and obtained results

Before running any of the scripts in the repo, make sure that the root folder is in  PYTHONPATH: 

_export PYTHONPATH=${PWD}_

## baseline model

Run from project root folder: _python src/trainer/ner-spacy-finetune.py -d raw_

For this pipeline, no preprocessing was done. Corpora was taken as is.

Existing taggets: _LOC, ORG, PER_

Newly added targets: _PRF, PRD, EVN, NAT, TIM, DOC, DAT_

Superflous tragets: _MISC_

Confusion matrix as CSV is given in models/finetuned_de_core_news_lg_raw/confusion.csv

Confusion matrix as PNG is given in models/finetuned_de_core_news_lg_raw/confusion.png

### Obtained preformance

| ents_p             | ents_r            | ents_f             |
| ------------------ | ----------------- | ------------------ |
| 0.7860262008733624 | 0.821917808219178 | 0.8035714285714286 |

| Entity | p                   | r                   | f                  |
|--------|---------------------|---------------------|--------------------|
| ORG    | 0.7142857142857143  | 0.8461538461538461  | 0.7746478873239436 |
| PRD    | 0.6666666666666666  | 0.47058823529411764 | 0.5517241379310345 |
| PER    | 0.8918918918918919  | 0.9428571428571428  | 0.9166666666666667 |
| PRF    | 0.7941176470588235  | 0.84375             | 0.8181818181818182 |
| EVN    | 0.8                 | 0.4                 | 0.5333333333333333 |
| LOC    | 0.8974358974358975  | 1                   | 0.945945945945946  |
| DAT    | 0.8571428571428571  | 0.8571428571428571  | 0.8571428571428571 |
| DOC    | 0.5                 | 0.3333333333333333  | 0.4                |
| TIM    | 0.8888888888888888  | 0.8888888888888888  | 0.8888888888888888 |
| NAT    | 0.42857142857142855 | 0.5                 | 0.4615384615384615 |

## fine-tuned model

python src/trainer/ner-spacy-finetune.py -d gold

A model pretrained for NER (_de_core_news_lg_) was taken as the basis.

Existing taggets: _LOC, ORG, PER_

Newly added targets: _PRF, PRD, EVN, NAT, TIM, DOC, DAT_

Superflous tragets: _MISC_

(Small) possiblity of catastrofical forgetting!

Confusion matrix as CSV is given in models/finetuned_de_core_news_lg_gold/confusion.csv

Confusion matrix as PNG is given in models/finetuned_de_core_news_lg_gold/confusion.png

### Obtained preformance

| ents_p             | ents_r             | ents_f             |
| ------------------ | ------------------ | ------------------ |
| 0.8410041841004184 | 0.8204081632653061 | 0.8305785123966942 |


| Entity | Precision          | Recall             | F-score             |
|--------|--------------------|--------------------|---------------------|
| PER    | 0.9459459459459459 | 0.9459459459459459 | 0.9459459459459459  |
| PRF    | 0.8421052631578947 | 0.8648648648648649 | 0.8533333333333334  |
| PRD    | 0.5263157894736842 | 0.5882352941176471 | 0.5555555555555555  |
| EVN    | 0.5                | 0.2222222222222222 | 0.30769230769230765 |
| LOC    | 0.9375             | 0.9375             | 0.9375              |
| NAT    | 1                  | 0.6666666666666666 | 0.8                 |
| ORG    | 0.8028169014084507 | 0.8382352941176471 | 0.8201438848920864  |
| TIM    | 1                  | 0.7777777777777778 | 0.8750000000000001  |
| DOC    | 0.6666666666666666 | 0.2857142857142857 | 0.4                 |
| DAT    | 0.875              | 1                  | 0.9333333333333333  |

## TRF model

run from project root folder:

The trainng datasets are provided in _data/docbin_. To (re)generate training data run: _python src/data/trf_dataset.py_

train a model from a cfg file: _python -m spacy train src/trainer/ner-tf.cfg --output ./models/trf_bert-base-german-cased_

<pre>
   E    #       LOSS TRANS...  LOSS NER  ENTS_F  ENTS_P  ENTS_R  SCORE
---  ------  -------------  --------  ------  ------  ------  ------
0       0          2639.10    372.31    0.98    0.72    1.54    0.01
40     200       214745.42  64104.06   72.09   69.40   75.00    0.72
80     400          576.13   3089.31   77.78   75.00   80.77    0.78
120     600          26.62   2894.61   77.47   75.09   80.00    0.77
160     800         772.50   2927.01   75.66   73.72   77.69    0.76
200    1000          33.62   2842.37   73.76   72.93   74.62    0.74
240    1200          63.04   2835.15   77.60   73.70   81.92    0.78
280    1400          30.08   2788.77   76.75   73.76   80.00    0.77
320    1600          12.80   2753.91   76.70   73.33   80.38    0.77
360    1800          12.49   2729.84   76.78   73.17   80.77    0.77
400    2000          58.89   2716.90   76.87   74.64   79.23    0.77
</pre>

# Annotate a document and extend the Person DBpedia class

We can use the fine-tuned model by using the _src/annotator/annotator.py_ script. 

It accepts two parameters if needed.  Both have default values set. 

To run the annotator you need to make sure there is a spaCy model in the model_path you want to use!
