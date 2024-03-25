import streamlit as st
from argparse import ArgumentParser
from models import *

def main(data_path, token):
    # page configuration
    st.set_page_config(
        page_title="biochat"
    )
    st.header("biochat")
    st.sidebar.header("Instructions")
    st.sidebar.info(
        '''
        This is a web app which allows you to interact with a chatbot specializing in biomedical information. 
        ''')
    st.sidebar.info('''First select a bot, then type a query in the text box and press enter
        to receive a response''')
    
    # bot selection
    st.sidebar.header("Bot Selection")
    name = st.sidebar.selectbox("Which bot would you like to use?",
                         ('General', 'Biomedical'))
    st.sidebar.info('''
                    General: A general chatbot for any kind of text prompt. Note the outputs of all models are text only.
                    ''')
    st.sidebar.info('''
                    Biomedical: A chatbot trained specifically for answering biomedical questions.
                    This model can better understand scientific language and terms as well as pull information from select papers.
                    ''')
    
    # initialize model
    generator = init_model(name=name, data_path=data_path, token=token)

    # display chat history
    # TO DO: handle different histories for different bots
    if "history" not in st.session_state:
        st.session_state.history = [] # list of dicts ('role': 'content')
    
    for message in st.session_state.history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # get input
    prompt = get_prompt()
    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.history.append({'role':'user', 'content': prompt}) # add to history
    
        # generate response
        if name=="Biomedical":
            db_path = "D:projects/biochat/clinical_trials/data/db"
            client = chromadb.PersistentClient(path=db_path)
            response = get_rag_response(prompt, generator, client)
        else:
            response = get_response(prompt, generator)
        with st.chat_message('assistant'):
            st.markdown(response)
        st.session_state.history.append({'role': 'assistant', 'content': response})

    
    
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', default='D:\\projects\\biochat\\model_caches')
    parser.add_argument('--huggingface_api_token', help='API token for access to models from HuggingFace')
    args = parser.parse_args()

    main(data_path=args.data_path, token=args.huggingface_api_token)