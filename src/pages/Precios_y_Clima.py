import streamlit as st
import os
from langchain.vectorstores import Pinecone, Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

#Crea la cadena de conversación
def get_conversation_chain(vstore):
    llm = ChatOpenAI(model='gpt-4', temperature=1)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vstore.as_retriever(),
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
    


    st.header("Chokobot :robot_face::chart::mostly_sunny:")
    #user_question = st.text_input("¿Cómo puedo ayudarte hoy?")
    user_question = st.chat_input("Escribe aquí...")   
    
    if user_question:
        handle_userinput(user_question)
    else: 
        st.info("Pulsa Nuevo Chat en el panel de la Izquierda y luego escribe algo para iniciar la conversación")    
    with st.sidebar:
        st.markdown("Precios y Clima en tiempo real")
        st.subheader("Pulsa 'Nuevo Chat' para iniciar conversación")
        if st.button("Nuevo Chat"):
            with st.spinner("Iniciando..."):
                                
                # tomar los datos para Chroma
                loader = WebBaseLoader(["https://www.accuweather.com/es/py/encarnacion/258296/daily-weather-forecast/258296",
                                        "https://www.accuweather.com/es/py/hohenau/258307/daily-weather-forecast/258307",
                                        "https://www.accuweather.com/es/py/asuncion/257012/daily-weather-forecast/257012",
                                        "https://www.bcp.gov.py/webapps/web/cotizacion/monedas",
                                        "https://www.cambioschaco.com.py/",
                                        "http://www.santaritacambios.com.py/",
                                        "https://es.investing.com/currencies/eur-pyg",
                                        "https://es.investing.com/currencies/usd-pyg",        
                                        "https://www.google.com/finance/quote/USD-PYG",
                                        "https://www.google.com/finance/quote/ARS-PYG",
                                        "https://www.google.com/finance/quote/BRL-PYG",
                                        "https://www.selinawamucii.com/es/perspectivas/precios/paraguay/trigo-bulgur/",
                                        "https://www.selinawamucii.com/es/perspectivas/precios/paraguay/trigo/",
                                        "https://www.selinawamucii.com/es/perspectivas/precios/paraguay/maiz/",
                                        "https://www.selinawamucii.com/es/perspectivas/precios/paraguay/arroz/",
                                        "https://www.selinawamucii.com/es/perspectivas/precios/paraguay/tomates/",
                                        "https://www.selinawamucii.com/es/perspectivas/precios/paraguay/cebollas/"
                                        ])
                document = loader.load()
    
                # Convertr en chunks
                text_splitter = RecursiveCharacterTextSplitter( chunk_size=400, chunk_overlap=100, length_function=len)
                document_chunks = text_splitter.split_documents(document)    
                # Crear el vector store
                vstore = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
                
                st.session_state.conversation = get_conversation_chain(vstore)              
                st.success("¡Listo!")             
        
if __name__ == '__main__':
    main()
