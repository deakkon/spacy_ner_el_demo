from pprint import pprint

import spacy
from definitions import ROOT_DIR
from spacy.language import Language
from SPARQLWrapper import SPARQLWrapper, JSON
from wikidata.client import Client
from spacy.language import Language

from src.annotator.queries import query_person

model_path = f"{ROOT_DIR}/models/de_core_news_lg_finetune"
example_text = """
    Wer ihm dabei zusah, wie er dribbelte und Zauberpässe spielte, fühlte, wie wichtig ihm dieser erste Titel mit Argentinien sein musste. Der Beleg dafür kam mit dem Abpfiff des Endspiels, der niederkniende Messi, die Mitspieler, die nicht ihren Sieg feierten, sondern seinen, die zu ihm liefen, ihn erdrückten. Ein weinender Messi, vor Glück, vor Erleichterung. Als Unvollendeter galt er vielen, weil er, der Mann aus dem Maradona-Land, zwar die Champions League gewonnen hatte und viele andere Pokale, aber die Argentinier nie zu einem Titel geführt hatte.
    """

queries = {
    "Person": query_person
}


class Annotator:

    """
    Use fine-tuned model and extend with DBPedia data.
    # todo update EL with https://microsoft.github.io/spacy-ann-linker/tutorial/remote_entity_linking/
    """

    def __init__(self, model_path=model_path):
        # config = Config().from_disk(f"{model_path}/config.cfg")
        # self.nlp = Language.from_config(config)
        self.available_linkers = ["dbpedia_spotlight", "opentapioca"]
        self.nlp = Language().from_disk(model_path)
        print(self.nlp.pipe_names)
        print("Loaded fine-tuned model")
        # self.nlp.add_pipe("dbpedia_spotlight",last=True, config={'language_code': 'de'})
        # todo add local DBpedia copy for http requests
        # nlp.add_pipe('dbpedia_spotlight', config={'dbpedia_rest_endpoint': 'http://localhost:2222/rest'})
        print("Added DBPedia linker")
        # alternatively, use wikidata lnker -> has an issue atm, need to inspect; requires local KB.
        # self.nlp.add_pipe("entityLinker", last=True)
        # print("Added Wikidata linker")
        # self.nlp.add_pipe('opentapioca')

        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.setReturnFormat(JSON)

    def fstr(self, template):
        return eval(f"f'{template}'")

    def link_dbpedia(self, text: str = None):
        self.switch_linker("dbpedia_spotlight")
        doc = self.nlp(text)
        for ent in doc.ents:
            dbpedia_raw_result = ent._.dbpedia_raw_result
            pprint(dbpedia_raw_result)
            types = [word.split(":")[1] for word in dbpedia_raw_result['@types'].split(",")
                     if word.startswith("Schema:")]
            wikiID = [word for word in dbpedia_raw_result['@types'].split(",")
                      if word.startswith("Wikidata:")]
            try:
                item = ent.kb_id_.split("/")[-1]
                uri = f"http://dbpedia.org/resource/{item}"
                query = queries[types[0].strip()].format(uri, uri)
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                pprint(results['results'])
            except KeyError:
                print(f"Currently not covering {types[0]}")
            except IndexError:
                print("We didnt get any required @types for URI. Check manually.")
            print("==============")

    def call_wikidata(self, text: str = None):
        self.switch_linker("opentapioca")
        doc = self.nlp(text)
        for span in doc.ents:
            print((span.text, span.kb_id_, span.label_, span._.description, span._.score))
            client = Client()
            entity = client.get(span.kb_id, load=True)
            pprint(entity)

    def switch_linker(self, linker):
        assert linker in self.available_linkers
        disable_linker = [x for x in self.available_linkers if x != linker][0]
        try:
            self.nlp.remove_pipe(disable_linker)
        except ValueError as err:
            pass
        self.nlp.add_pipe(linker, linker)
        print(f"Using {linker} with the pipeline: {self.nlp.pipe_names}")


if __name__ == "__main__":
    model = Annotator()
    model.link_dbpedia(example_text)


