import torch
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector
from chromadb import EmbeddingFunction, Documents, Embeddings
from bs4 import BeautifulSoup
import requests
import os
from tqdm import tqdm

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
    data_path = "D:\\projects\\biochat\\pubmed\\publications"

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
    for lab in tqdm(labs):
        url = "https://uofuhealth.utah.edu/huntsman/labs/{}/publications".format(lab)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        div = soup.find('div', class_='coh-wysiwyg')

        texts = []
        title = False
        for tag in div.find_all('p'):
            if lab in {'onega', 'young', 'mcmahon', 'edgar', 'johnson'}: # plain text
                text = tag.text.strip()
                is_header = (text.split()[0] in {'View', 'Current', 'Full', 'If', 'These'}) or ((text.split()[0] + text.split()[1]) in {'Toview', 'Labmembers'})

            elif lab == 'ayer': # handle duplicate titles
                content = tag.text.strip()
                is_header = (content.split()[0] in {'View', 'Current', 'Full', 'If', 'These'}) or ((content.split()[0] + content.split()[1]) in {'Toview', 'Labmembers'})
                if not is_header:
                    text = content.split('\n')[1]

            elif tag.find('a'): # standard format
                text = tag.text.strip()
                is_header = (text.split()[0] in {'View', 'Current', 'Full', 'If', 'These'}) or ((text.split()[0] + text.split()[1]) in {'Toview', 'Labmembers'})
            
            if not is_header:
                texts.append(text.replace("\n", ""))

        path = os.path.join(data_path, "{}.txt".format(lab.replace("-", "_")))
        with open(path, "w", encoding='utf-8') as file:
            for t in texts:
                file.write(str(t) + "\n")

    for lab in tqdm(unique_urls.keys()):
        url = unique_urls[lab]
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if lab == "anderson":
            div = soup.find_all('div', class_="sqs-html-content")[1]
            texts = []
            group = []
            for p in div.find_all('p'):
                if p.text.strip():
                    group.append(p.text.strip())
                elif group: # if p is empty and group is non-empty
                    texts.append(". ".join(group))
                    group = []
            if group: # add last group
                texts.append(". ".join(group))

            path = os.path.join(data_path, "anderson.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")
        
        if lab == "basham":
            div = soup.find('div', id='Containerc24vq')
            texts = []
            for p in div.find_all('p', class_='font_7'):
                text = p.text.strip()
                texts.append(text)
            
            path = os.path.join(data_path, "basham.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")

        if lab == "buckley":
            div = soup.find_all('div', class_='col-sm-10 col-sm-offset-1')[1]
            texts = []
            for p in div.find_all('p')[:-1]:
                if p.text.strip() != "(Selected Publications Since 2010)":
                    text = p.text.strip()
                    texts.append(text.lstrip('-\t').strip())
            path = os.path.join(data_path, "buckley.txt")
            with open(path, 'w', encoding='utf-8') as file:
                for text in texts:
                    file.write(str(text) + "\n")
                

                

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