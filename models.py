from transformers import pipeline
import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

def init_model(name, data_path, token):
    path = os.path.join(data_path, name)
    download = (len(os.listdir(path))==0) # download if cache directory is empty
    print(path, download)
    if name == "Biomedical":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt", cache_dir=path, force_download=download)
        model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt", cache_dir=path, force_download=download)
    
    if name == "General":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", cache_dir=path, force_download=download, token=token)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", cache_dir=path, force_download=download, token=token)
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator


def get_response(prompt, generator):
    message = generator(
        prompt,
        max_length=200,
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