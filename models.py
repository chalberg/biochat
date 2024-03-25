from transformers import pipeline
import streamlit as st
import chromadb
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from data_utils import PubMedBertBaseEmbeddings

def init_model(name, data_path, token):
    path = os.path.join(data_path, name)
    download = (len(os.listdir(path))==0) # download if cache directory is empty
    if name == "Biomedical":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", cache_dir=path, force_download=download)
        model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt", cache_dir=path, force_download=download)
    
    if name == "General":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", cache_dir=path, force_download=download, token=token)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", cache_dir=path, force_download=download, token=token)
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator


def get_rag_response(prompt, generator, client):
    # query vector db
    model = PubMedBertBaseEmbeddings(input=[prompt])
    collection = client.get_collection(name='clinicaltrials', embedding_function=model)
    texts = collection.query(query_texts=[prompt], n_results=3)["documents"][0]
    # generate from augmented prompt
    aug_prompt = f'''{prompt}?
                Please answer this question using the following information;
                {texts[0]} 
                {texts[1]}
                {texts[2]}
                '''
    message = generator(
        aug_prompt,
        max_length=600,
        num_return_sequences=1,
        do_sample=False
    )
    message[0]['generated_text']

def get_response(prompt, generator):
    message = generator(
        prompt,
        max_length=600,
        num_return_sequences=1,
        do_sample=False
    )
    return message[0]['generated_text']

def get_prompt():
    # TO DO
    # - prompt augmentation based on model
    input_text = st.chat_input()
    prompt = input_text
    return prompt