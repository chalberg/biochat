from transformers import pipeline
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_response(prompt):
    message = generator(
        prompt,
        max_length=200,
        num_return_sequences=1,
        do_sample=False
    )
    return message[0]['generated_text']

def get_prompt():
    input_text = st.text_input("You: ","", key="input")
    return input_text 