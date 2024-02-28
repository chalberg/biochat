import torch
from transformers import AutoTokenizer, AutoModel
import mysql.connector
from chromadb import EmbeddingFunction, Documents, Embeddings

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