import streamlit as st 
from streamlit_chat import message
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Loading the model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.Q4_0.gguf",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

st.title("Chat with Crystal Quantum Shield 🤖")

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    # Use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load the data using pandas for analysis and graphing, limit to 10 rows
    df = pd.read_csv(tmp_file_path, nrows=10)

    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Column 1: Data Analysis and Visualization
    with col1:
        st.write("## Data Preview")
        st.write(df.head())

        # Display basic statistics
        st.write("## Data Statistics")
        st.write(df.describe())

        # Data visualization
        st.write("## Data Visualization")

        # Example: Pairplot using Seaborn for numerical columns
        if st.checkbox("Show Pairplot", key='pairplot'):
            st.write("Pairplot of numerical columns:")
            pairplot_fig = sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
            st.pyplot(pairplot_fig)

        # Example: Correlation heatmap using Seaborn
        if st.checkbox("Show Correlation Heatmap", key='heatmap'):
            st.write("Correlation Heatmap:")
            corr = df.corr()
            heatmap_fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(heatmap_fig)

    # Column 2: Chatbot Interface
    with col2:
        st.write("## Chatbot Interface")

        # Load data for chatbot
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        data = loader.load()
        
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)
        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        def conversational_chat(query):
            result = chain.invoke({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]
        
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]
        
        # Container for the chat history
        response_container = st.container()
        # Container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
