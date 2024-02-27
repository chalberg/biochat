import chromadb
from pytrials.client import ClinicalTrials
from argparse import ArgumentParser
import torch
import os
from tqdm import tqdm
from data_utils import PubMedBertBaseEmbeddings

def init_db(test, embedding_model):
    if test:
        data_path = "D:projects/biochat/clinical_trials/test/raw"
        db_path = "D:projects/biochat/clinical_trials/test/db"

        ids = []
        docs = []
        for file in os.listdir(data_path):
            path = os.path.join(data_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                id = file.split('.')[0]
                ids.append(id)
                docs.append(text)

        # make / update vectordb
        if embedding_model == "neuml/pubmedbert-base-embeddings":
            model = PubMedBertBaseEmbeddings(input=docs)
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name='test', embedding_function=model)
        collection.add(ids=ids, documents=docs)
        # collection info
        print(collection.count())
        print(collection.peek())

def update_db(test):
    pass

def get_clinical_data(test):
    if test:
        # load sample NCTIds
        with open('data/sample_studies.txt', 'r') as sample_trials:
            trials = sample_trials.read()
            ids = list(trials.splitlines())

        ct = ClinicalTrials()
        #fields = ['NCTId','BriefTitle','BriefSummary','Condition','OverallStatus','Phase','StdAge','Keyword','StudyType','LocationFacility','InterventionName','InterventionType','InterventionDescription','PrimaryOutcomeMeasure']
        fields = ['NCTId', 'BriefSummary']
        data_path = "D:projects/biochat/clinical_trials/test/raw"
        for id in tqdm(ids):
            study = ct.get_study_fields(search_expr='{}'.format(id),
                                        fields=fields,
                                        fmt='csv',
                                        max_studies=1)
            text = study[1][2]
            path = os.path.join(data_path, "{}.txt".format(id))
            with open(path, "w", encoding='utf-8') as file:
                file.write(text)
            

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument('--update_db', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--embedding_model', default="neuml/pubmedbert-base-embeddings", help="Embedding model from Hugging Face to be used")

    args = parser.parse_args()

    init_db(test=args.test, embedding_model=args.embedding_model)
    #get_clinical_data(test=args.test)