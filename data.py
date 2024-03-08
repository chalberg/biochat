import chromadb
import mysql.connector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pytrials.client import ClinicalTrials
from argparse import ArgumentParser
import os
from tqdm import tqdm
from data_utils import *

def init_db(test, embedding_model, db_user, db_pswd):
    if embedding_model=="neuml/pubmedbert-base-embeddings":
        chunk_size = 512

    if test:
        data_path = "D:projects/biochat/clinical_trials/test/raw"
        db_path = "D:projects/biochat/clinical_trials/test/db"

        # create reference db
        conn, cursor = init_db_client(user=db_user, pswd=db_pswd)
        cursor.execute("CREATE DATABASE IF NOT EXISTS biochat;")
        conn.commit()
        cursor.close()
        conn.close()

        # create test table in biochat db
        conn, cursor = init_db_client(user=db_user, pswd=db_pswd, db="biochat")
        table_query = """
            CREATE TABLE IF NOT EXISTS test (
                id VARCHAR(48) PRIMARY KEY,
                BriefSummary VARCHAR({chunk_size})
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
    
    else:
        # create reference db
        conn, cursor = init_db_client(user=db_user, pswd=db_pswd)
        cursor.execute("CREATE DATABASE IF NOT EXISTS biochat;")
        conn.commit()
        cursor.close()
        conn.close()

        # create clinicaltrials table in biochat db
        conn, cursor = init_db_client(user=db_user, pswd=db_pswd, db="biochat")
        cursor.execute("DROP TABLE IF EXISTS clinicaltrials")
        cursor.fetchall()
        table_query = """
            CREATE TABLE clinicaltrials (
                id VARCHAR(48) PRIMARY KEY,
                BriefTitle TEXT,
                BriefSummary TEXT,
                `Condition` TEXT,
                OverallStatus TEXT,
                Phase TEXT,
                StdAge TEXT,
                Keyword TEXT,
                StudyType TEXT,
                LocationFacility TEXT,
                InterventionName TEXT,
                InterventionType TEXT,
                InterventionDescription TEXT,
                PrimaryOutcomeMeasure TEXT
            );
            """.format(chunk_size=chunk_size)
        cursor.execute(table_query)
        conn.commit()
        cursor.close()
        conn.close()

def update_db(test, data, embedding_model, db_user, db_pswd):
    # data = {id1: [var1, var2, ..., varn],
    #         id2: [var1, var2, ..., varn],
    #          ... }
    conn, cursor = init_db_client(user=db_user, pswd=db_pswd, db="biochat")
    if test:
        db_path = "D:projects/biochat/clinical_trials/test/db"

        query = """SELECT id FROM test"""
        cursor.execute(query)
        ids = cursor.fetchall()
        for id in data.keys():
            if id not in set(ids):
                fields = data[id]
                ids, docs = split_clinicaltrials_data(embedding_model=embedding_model, data=(id, fields))
                for i, t in zip(ids, docs):
                    insert_query = """INSERT INTO test VALUES (%s, %s)"""
                    cursor.execute(insert_query, (i, t))
                
                # add to vector db
                if embedding_model == "neuml/pubmedbert-base-embeddings":
                    model = PubMedBertBaseEmbeddings(input=docs)
                client = chromadb.PersistentClient(path=db_path)
                collection = client.get_or_create_collection(name='test', embedding_function=model)
                collection.add(ids=ids, documents=docs)
        conn.commit()
        cursor.close()
        conn.close()
    
    else:
        db_path = "D:projects/biochat/clinical_trials/data/db"

        conn, cursor = init_db_client(user=db_user, pswd=db_pswd, db="biochat")
        query = """SELECT id FROM clinicaltrials"""
        cursor.execute(query)
        result = cursor.fetchall()
        nctids = set([str(i[0])for i in result])

        vector_ids = []
        vector_docs = []
        num_new = 0
        for id in data.keys():
            if id not in nctids:
                # insert raw data into MySQL db
                values = tuple(data[id])
                insert_query = """
                    INSERT INTO clinicaltrials (
                        id, BriefTitle, BriefSummary, `Condition`, OverallStatus,
                        Phase, StdAge, Keyword, StudyType, LocationFacility,
                        InterventionName, InterventionType, InterventionDescription,
                        PrimaryOutcomeMeasure)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                fields = (id,)+values
                cursor.execute(insert_query, fields)

                # chunk for vector embeddings
                i, d = split_clinicaltrials_data(embedding_model=embedding_model, data=fields)
                vector_ids.extend(i)
                vector_docs.extend(d)
                num_new += 1

        # update vectordb
        if vector_ids:
            if embedding_model == "neuml/pubmedbert-base-embeddings":
                model = PubMedBertBaseEmbeddings(input=vector_docs)
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection(name='clinicaltrials', embedding_function=model)
            collection.add(ids=vector_ids, documents=vector_docs)
            conn.commit()

        print("Added {} new studies to the database".format(num_new))
        cursor.close()
        conn.close()

def get_clinicaltrials_data(search_exp, test):
    if test:
        # load sample NCTIds
        with open('data/sample_studies.txt', 'r') as sample_trials:
            trials = sample_trials.read()
            ids = list(trials.splitlines())

        ct = ClinicalTrials()
        data_path = "D:projects/biochat/clinical_trials/test/raw"
        for id in tqdm(ids):
            study = ct.get_study_fields(search_expr='{}'.format(id),
                                        fields=['NCTId', 'BriefSummary'],
                                        fmt='csv',
                                        max_studies=1)
            text = study[1][2]
            path = os.path.join(data_path, "{}.txt".format(id))
            with open(path, "w", encoding='utf-8') as file:
                file.write(text)
    else:
        # pull data from ClinicalTrials.gov
        ct = ClinicalTrials()
        fields = ['NCTId','BriefTitle','BriefSummary','Condition','OverallStatus',
                  'Phase','StdAge','Keyword','StudyType','LocationFacility',
                  'InterventionName','InterventionType','InterventionDescription','PrimaryOutcomeMeasure']
        studies = ct.get_study_fields(search_expr="{}".format(search_exp),
                                   fields=fields,
                                   max_studies=100,
                                   fmt='csv')
        data = {}
        for i in range(1, len(studies)):
            data[studies[i][1]] = studies[i][2:]
        
        return data
        
            
if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument('--update_db', action='store_true')
    parser.add_argument('--init_db', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--embedding_model', default="neuml/pubmedbert-base-embeddings", help="Embedding model from Hugging Face to be used")
    parser.add_argument('--db_pswd', help="Pasword for the MySQL database storing context data")
    parser.add_argument('--db_user', default="root", help="User for the MySQL database for storing context data")
    parser.add_argument('--search', default=None, help='Search term for querying ClinicalTrials.gov')
    parser.add_argument('--scrape_papers',  action='store_true')
    args = parser.parse_args()

    conn, cursor = init_db_client(user=args.db_user, pswd=args.db_pswd, db="biochat")
    query = "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'clinicaltrials'"
    cursor.execute(query)
    result = cursor.fetchone()[0] == 1
    if (not result) or (args.init_db):
        print("Initializing database ...")
        init_db(test=args.test, embedding_model=args.embedding_model, db_pswd=args.db_pswd, db_user=args.db_user)
    
    if args.search is not None:
        print("Pulling data from ClinicalTrials.gov ...")
        data = get_clinicaltrials_data(search_exp=args.search, test=args.test)

    if args.update_db:
        print("Updating databases ...")
        update_db(test=args.test, data=data, embedding_model=args.embedding_model, db_user=args.db_user, db_pswd=args.db_pswd)

    if args.scrape_papers:
        scrape_huntsman_publications()