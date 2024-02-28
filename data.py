import chromadb
import mysql.connector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pytrials.client import ClinicalTrials
from argparse import ArgumentParser
import os
from tqdm import tqdm
from data_utils import PubMedBertBaseEmbeddings, init_db_client

def init_db(test, embedding_model, db_user, db_pswd):
    if test:
        data_path = "D:projects/biochat/clinical_trials/test/raw"
        db_path = "D:projects/biochat/clinical_trials/test/db"

        if embedding_model=="neuml/pubmedbert-base-embeddings":
            chunk_size = 512

        # create reference db
        conn, cursor = init_db_client(user=db_user, pswd=db_pswd)
        cursor.execute("CREATE DATABASE IF NOT EXISTS biochat;")
        conn.commit()
        cursor.close()
        conn.close()

        ## change db data path
        #conn, cursor = init_db_client(user=db_user, pswd=db_pswd, db="biochat")
        #query = """SET GLOBAL datadir='{path}';""".format(path=db_path) # change data directory
        #print(query)
        #cursor.execute(query)
        #conn.commit()
        #cursor.close()
        #conn.close()
        #os.system('sudo systemctl restart mysql') # restart MySQL to update data dir

        # create test table in biochat db
        conn, cursor = init_db_client(user=db_user, pswd=db_pswd, db="biochat")
        table_query = """
            CREATE TABLE IF NOT EXISTS test (
                id VARCHAR(48) PRIMARY KEY,
                BriefDescription VARCHAR({chunk_size})
            )
            """.format(chunk_size=chunk_size)
        cursor.execute(table_query)
        conn.commit()

        # read clinicaltrials data
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        ids = []
        docs = []
        print("Creating database ...")
        for file in tqdm(os.listdir(data_path)):
            path = os.path.join(data_path, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                texts = splitter.split_text(content)
                for i, t in enumerate(texts):
                    id = file.split('.')[0]
                    id = id+"_"+str(i)
                    ids.append(id)
                    docs.append(t)
                    
                    # update MySQL db
                    insert_query = """INSERT INTO test (id, BriefDescription) VALUES (%s, %s)"""
                    cursor.execute(insert_query, (id, t))
        
        conn.commit() # commit changes to db
        cursor.close()
        conn.close()
        
        # make / update vectordb
        print("Creating vectorized database ...")
        if embedding_model == "neuml/pubmedbert-base-embeddings":
            model = PubMedBertBaseEmbeddings(input=docs)
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name='test', embedding_function=model)
        collection.add(ids=ids, documents=docs)
        print("Done!")

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
    parser.add_argument('--db_pswd', help="Pasword for the MySQL database storing context data")
    parser.add_argument('--db_user', default="root", help="User for the MySQL database for storing context data")
    args = parser.parse_args()

    init_db(test=args.test, embedding_model=args.embedding_model, db_pswd=args.db_pswd, db_user=args.db_user)
    if args.update_db:
        update_db(test=args.test)

    #get_clinical_data(test=args.test)