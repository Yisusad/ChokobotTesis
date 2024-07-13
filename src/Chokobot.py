import streamlit as st
import os
from utils import *
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_ENV = st.secrets['PINECONE_ENV']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
INDEX_NAME = st.secrets['PINECONE_INDEX_NAME']

#Crea la cadena de conversación
def get_conversation_chain(vstore):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever(search_type='similarity', search_kwargs={'k': 10}),
        memory=memory
    )
    return conversation_chain

#Maneja la entrada del usuario
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

#Función principal
def main():
    load_dotenv()
    st.set_page_config(page_title="Chokobot",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  

    st.header("Chokobot :robot_face::seedling:")
    #user_question = st.text_input("¿Cómo puedo ayudarte hoy?")
    
    user_question = st.chat_input("Escribe aquí...")     

    if user_question:
        handle_userinput(user_question)
    else: 
        st.info("Pulsa Nuevo Chat en el panel de la Izquierda y luego escribe algo para iniciar la conversación")    
    with st.sidebar:
        st.markdown("Información sobre agricultura")
        st.subheader("Pulsa 'Nuevo Chat' para iniciar conversación")
        if st.button("Nuevo Chat"):
            with st.spinner("Iniciando..."):
                m = 'ok'
                # Crear la cadena(chain) de Conversación
                os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
                embeddings = OpenAIEmbeddings()
                vstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)  
                st.session_state.conversation = get_conversation_chain(vstore)              
                st.success("¡Listo!")             
      
if __name__ == '__main__':
    main()
