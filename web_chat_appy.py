import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import requests
import re
from bs4 import BeautifulSoup
from streamlit_modal import Modal           
import langchain
langchain.verbose = False
@st.cache_resource
def init_connection():
# Create a new client and connect to the server
    return MongoClient(st.secrets['uri'], server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client = init_connection()
    client.admin.command('ping')
    db=client["uri_chat"]
    collection=db['chats']
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

openai_api_key =st.secrets['OPENAI_API_KEY']
def new_button():                                                                                                           # Call-back function for New Chat Button
    st.session_state.is_chatting= True
    st.session_state.messages = []
    st.session_state.clicked = True
        
def old_data():                                                                                                             # Call-back function for Displaying old conversations

        for i,j in enumerate(db['c1'].find({}, {'_id': False})):
            with st.expander((f"chat {i+1}")):    
                st.write(j)
def insert_data(dic):                                                                                                       # Call-back function to insert data on the MongoDB databases
    try:
        db['c1'].insert_one(dic)
        print("Pass")
                            
    except:
        print("fail")
            
def clear_old_data():                                                                                                       # Call-back function for clearing all the old conversations in the Database
        db['c1'].delete_many({})

def get_soup(url):                                                                                                          
    page = requests.get(url)                                                                                                # Page content from Website URL
    soup = BeautifulSoup(page.content , 'html.parser')                                                                      # Parse HTML content
    return soup
def clean_content(text):                                                                                                    # Erasing reference numbers
    text = re.sub("\[\d+\]", "" , text)
    text = text.replace("[edit]", "")
    return text
def get_paragraph_text(p):                                                                                                  # Extracting data from p bodies with their child bodies
    paragraph_text = ''
    for tag in p.children:
        paragraph_text = paragraph_text + tag.text
    
    return paragraph_text
def get_clean_extract(url):                                                                                                 # Cleaning all the extracted data
    soup = get_soup(url) 
    headers = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    clean_extract = []
    for tag in soup.find_all():
        if tag.name in headers and tag.text != 'Contents':
            # We try to find all paragraphs after it
            p = ''
            # loop through the next elements
            for ne in tag.next_elements:
                if ne.name == 'p':
                    p = p + get_paragraph_text(ne)
                if ne.name in headers:
                    break
            if p != '':
                section = [clean_content(tag.text), tag.name, clean_content(p)]
                clean_extract.append(section)
        
    return clean_extract
    
if 'clicked' not in st.session_state:                                                                                       # Creating clicked variable in session state
    st.session_state.clicked = False


if 'chat_history' not in st.session_state:                                                                                  # Creating a list -chat history  in session state if not existed
    st.session_state.chat_history=[]

if "messages" not in st.session_state:                                                                                      # Creating messages list in session state if not existed
    st.session_state.messages = []

for message in st.session_state["messages"]:
    if message["role"] == "user":
                                                                                                                            # Display a user message in the chat interface
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
                                                                                                                            # Display an assistant message in the chat interface
        with st.chat_message("assistant"):
            st.markdown(message["content"])

with st.sidebar:
   
    st.title("📃Interactive Web URL Chat App")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    ''')
    
    st.button("New Chat",key='start_a_new_Chat',help="click to start new chat",on_click=new_button)                         # Creating New Chat button with callback function
          
    with st.form("form",clear_on_submit=False):                                                                             # Creating Old Conversations form which contains all the previous chats
        st.title("Old Conversions")
        st.form_submit_button("Old Conversations",on_click=old_data())
    st.button("Clear Old Data",on_click=clear_old_data)                                                                     # Creating Clear Old chats button with call function
    
    st.markdown(' ## For Reference📑📚-')
    st.write('''-  [Streamlit](https://streamlit.io/)
            -  [LangChain](https://python.langchain.com/)
            -  [OpenAI](https://platform.openai.com/docs/models) LLM model
            ''')
    


def main():
     
    st.header("Chat with Wikipedia Web URL file 💬")

    load_dotenv()                                                                                                          # Setting up an envirnment for openai with authentication key
                                                            
    url = st.text_input("Enter the url🌐..")                                                                               # To upload a URL to the app
    if url :
        modal = Modal("Demo Modal",key='demo')
        open_modal = st.button("Quick look to the url?")
        if open_modal:
            modal.open()

        if modal.is_open():
            with modal.container():
                st.components.v1.iframe(src=url, width=None, height=550, scrolling=True)                                  # Creating scrollable (Modal) frame for url data
                
                  
        extracted_text=get_clean_extract(url)                                                                             # Getting all the data from the url
        content_title=extracted_text[0][0]                                                                                # Extracting the title of the URL
        st.write(content_title)                                                                                           
        text_spiltter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=400,length_function=len)              # Defining the parameters for text splitting
        chuncks=text_spiltter.split_text(text=extracted_text[0])                                                          # Creating chuncks from the splitted text
    

                                                            
                                                                           
        
        if os.path.exists(f"{content_title}.pk1"):                                                                            # If the url  is already used, then use the same previous embeddings for cost effeciency
            with open(f'{content_title}.pk1','rb') as file:
                Vector=pickle.load(file)
        else:
            embeddings=OpenAIEmbeddings()                                                                  
            Vector=FAISS.from_texts(chuncks,embedding=embeddings)                                                          # If a new url is used, then create new embeddings 
            with open(f'{content_title}.pk1','wb') as file:
                pickle.dump(Vector,file)
        
        

        query = st.chat_input(placeholder="Ask question from your url page 🔎:")                                           # Questions regarding the url
        
        if st.session_state.start_a_new_Chat:                                                                              # When New Chat button is pressed.. a new session will be created
            st.session_state.is_chatting = True
            st.session_state.messages = []
            st.session_state.chat_history=[]
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)                                 # This memory allows for storing messages and then extracts the messages in a variable
        
        if query:
            chat_history = []
            with st.chat_message("user"):
                st.write("Please wait..baking your results...⌛⌛")                                         
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})                                           # Storing the user data in session state list variable messages
                                                                                                                            # For generating prompts for language models
            custom_template = """                                                                                                      
            Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
            At the end of the standalone question, add this 'Answer the question in English language.'
            If you do not know the answer, reply with 'I am sorry'.
            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:
            Remember to greet the user with 'hi welcome to the PDF chatbot, how can I help you?' if the user asks 'hi' or 'hello.'
            """                                                                                                                                 
            
            CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


            docs=Vector.similarity_search(query=query,k=3)                                                                 # Checking similarites in the vectorspace and the query
            llm=ChatOpenAI()                                                                                               # Model for Conversational Chat
                                                                                                                            # Combining model, chat history as memory with the generated prompt
            chain=ConversationalRetrievalChain.from_llm(llm,Vector.as_retriever(),condense_question_prompt=CUSTOM_QUESTION_PROMPT,memory=memory)

            with get_openai_callback() as callback:
                response=chain({"question": query, "chat_history": chat_history})                                          # Generating Response from the llm about the query with best match from the simalarity search
                #  print(callable)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
                
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})                         # Storing the assistant generated data in session state list variable messages
            
            
            
                
            st.session_state.chat_history.append((query, response['answer']))                                              # Adding the assistant data to the chat history
            
            
            dic=dict(st.session_state.chat_history)                                                                        # Creating dictionary for adding the data to Database
            st.button("End Chat",on_click=insert_data,args=(dic,))                                                         # Creating of End chat for storing the last generated data to Database
                
                
                            

if __name__ == '__main__':
    main()
