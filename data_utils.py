import torch
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
from chromadb import EmbeddingFunction, Documents, Embeddings
from bs4 import BeautifulSoup
import requests

def init_db_client(user, pswd, db=None):
    if db:
        conn = mysql.connector.connect(
            host="localhost",
            user=user,
            password=pswd,
            database=db
        )
        return conn, conn.cursor()
    else:
         conn = mysql.connector.connect(
              host="localhost",
              user=user,
              password=pswd,
         )
         return conn, conn.cursor()

def split_clinicaltrials_data(embedding_model, data):
        # data = (id, var1, var2, ..., varn)
        if embedding_model == "neuml/pubmedbert-base-embeddings":
            chunk_size = 512
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

        combined = "".join(data[1:]) # combine values into single string

        docs, ids = [], []
        id = data[0]
        texts = splitter.split_text(combined)
        for i, t in enumerate(texts):
            ids.append(id+"_"+str(i))
            docs.append(t)
        return ids, docs

def scrape_huntsman_publications():
    data_path = 'D:\projects\biochat\pubmed\publications'

    # get lab names
    labs = ['ayer', 'beckerle', 'bernard', 'cairns-lab', 'camp', 'chandrasekharan',
            'cheshier', 'curtin', 'doherty', 'edgar', 'evason', 'gaffney', 'gertz',
            'graves', 'grossman', 'hashibe', 'holmen', 'hu-lieskovan', 'jensen-lab',
            'johnson', 'kb-jones-lab', 'kaphingst', 'kepka', 'kinsey', 'kirchhoff',
            'mcmahon', 'mendoza', 'mooney', 'neklason', 'onega', 'schiffman', 'snyder',
            'spike', 'stewart', 'suneja', 'tavtigian', 'ullman', 'vanbrocklin', 'varley',
            'young', 'zhang']

    unique_urls = {'anderson': 'https://www.joshandersenlab.com/publications',
                   'basham': 'https://www.bashamlab.com/publications',
                   'buckley': 'https://buckleylab.org/',
                   'allie-grossmann': 'https://medicine.utah.edu/pathology/research-labs/allie-grossmann',
                   'torres': 'https://www.judsontorreslab.org/publications',
                   'myers': 'http://www.myerslab.org/publications.html',
                   'tan': 'http://tanlab.org/papers.html',
                   'welm-a': 'https://pubmed.ncbi.nlm.nih.gov/?term=Welm-A&sort=date&sort_order=asc&format=pubmed&size=100',
                   'welm-b': 'https://pubmed.ncbi.nlm.nih.gov/?term=Welm-B&sort=date&format=pubmed&size=100',
                   'wu': 'https://uofuhealth.utah.edu/huntsman/labs/wu/publications',
                   'ulrich' : 'https://uofuhealth.utah.edu/huntsman/labs/ulrich/publications'}

    # scrape publications from standard page format
    for lab in labs:
        url = "https://uofuhealth.utah.edu/huntsman/labs/{}/publications".format(lab)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        div = soup.find('div', class_='coh-wysiwyg')
        tags = div.find_all('p')
        count = 1
        
        for tag in tags:
            if lab in {'onega', 'young', 'mcmahon', 'edgar'}:
                text = tag.text
                is_header = (text.split()[0] in {'View', 'Current', 'Full', 'If', 'These'}) or ((text.split()[0] + text.split()[1]) in {'Toview', 'Labmembers'})
                if not is_header:
                    count += 1

            elif tag.find('a'):
                text = tag.text
                is_header = (text.split()[0] in {'View', 'Current', 'Full', 'If', 'These'}) or ((text.split()[0] + text.split()[1]) in {'Toview', 'Labmembers'})
                if not is_header:
                    count += 1

    for lab in unique_urls.keys():
        url = unique_urls[lab]
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # handle separately for each page
         
     

class PubMedBertBaseEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # generate text embeddings
        tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
        model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
        inputs = tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            output = model(**inputs)
        embeddings = self.meanpooling(output, inputs['attention_mask'])
        return embeddings.tolist()
    
    def meanpooling(self, output, mask):
        embeddings = output[0] # First element of model_output contains all token embeddings
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)