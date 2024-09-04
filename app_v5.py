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

    # Load the full data using pandas for analysis and graphing
    df_full = pd.read_csv(tmp_file_path)

    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Column 1: Data Analysis and Visualization
    with col1:
        st.write("## Data Preview")
        st.write(df_full)  # Show the full DataFrame

        # Display basic statistics
        st.write("## Data Statistics")
        st.write(df_full.describe(include='all'))  # Include all data types for descriptive stats

        # Data visualization
        st.write("## Data Visualization")

        # Example: Countplot for 'Level'
        if st.checkbox("Show Level Countplot", key='level_countplot'):
            st.write("Countplot of Levels:")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(x='Level', data=df_full, ax=ax)
            st.pyplot(fig)

        # Example: Time series plot for 'Timestamp'
        if st.checkbox("Show Timestamp Time Series", key='timestamp_timeseries'):
            st.write("Time Series of Events:")
            df_full['Timestamp'] = pd.to_datetime(df_full['Timestamp'])
            fig, ax = plt.subplots(figsize=(12, 6))
            df_full.groupby(df_full['Timestamp'].dt.date).size().plot(ax=ax)
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Events')
            ax.set_title('Number of Events per Day')
            st.pyplot(fig)

        # Example: Ring chart for failed vs successful logins
        if st.checkbox("Show Login Success vs Failure", key='login_ring_chart'):
            st.write("Login Success vs Failure Ring Chart:")
            login_counts = df_full['Status'].value_counts()
            labels = login_counts.index
            sizes = login_counts.values
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                                              wedgeprops=dict(width=0.3))
            ax.set_title('Failed vs Successful Logins')
            st.pyplot(fig)

    # Column 2: Chatbot Interface
    with col2:
        st.write("## Chatbot Interface")

        # Load only 10 rows for the chatbot
        df_chatbot = pd.read_csv(tmp_file_path, nrows=10)
        
        # Save the limited data for the chatbot
        with tempfile.NamedTemporaryFile(delete=False) as temp_chatbot_file:
            df_chatbot.to_csv(temp_chatbot_file.name, index=False)
            chatbot_tmp_file_path = temp_chatbot_file.name

        # Load data for chatbot
        loader = CSVLoader(file_path=chatbot_tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
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
