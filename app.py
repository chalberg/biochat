import streamlit as st
from streamlit_chat import message
from models import get_prompt, get_response

def main():
    st.set_page_config(
        page_title="biochat"
    )
    st.header("biochat")
    st.sidebar.header("Instructions")
    st.sidebar.info(
        '''This is a web application that allows you to interact with an 
        EHR knowledge graph, ask biomedical questions or general questions. 
        '''
        )
    st.sidebar.info('''Enter a query in the text box and press enter
        to receive a response''')
    
    st.sidebar.info('''The app is under active development. 
        There are several issues that needs to be fix''')
    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    
    if 'past' not in st.session_state:
        st.session_state['past'] = []
        
    
    user_input = get_prompt()
    
    if user_input:
        output = get_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    
if __name__=="__main__":
    main()