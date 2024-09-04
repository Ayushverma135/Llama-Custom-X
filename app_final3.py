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

# Define default paths for CSV files
DEFAULT_CSV_PATH_CHATBOT = 'securityserver3.csv'
# DEFAULT_CSV_PATH_DATASET1 = 'securityserver3.csv'
DEFAULT_CSV_PATH_DATASET2 = 'log.csv'
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

# Centered title with Markdown and HTML
st.markdown(
    """
    <div style='text-align: center;'>
        <span style='font-size:36px; font-family:Courier New; font-weight:bold;'>C.Q.S</span>
        <br>
        <span style='font-size:36px; font-family:Courier New; font-weight:bold;'>Crystal Quantum Shield 🤖</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Remove margins from the sides
st.markdown(
    """
    <style>
    .main {
        margin-left: 0;
        margin-right: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Column 1: Chatbot Interface
st.write("## Cyber Assistant")

# Use st.expander for the chatbot interface to make it more spacious and collapsible
with st.expander("Open Cyber Assistant", expanded=True):
    # Load only 10 rows for the chatbot
    df_chatbot = pd.read_csv(DEFAULT_CSV_PATH_CHATBOT, nrows=10)
    
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
        st.session_state['generated'] = ["Hello! Ask me anything about the data 🤗"]

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

# Add a separator line
st.write("---")

# Data Preview and Visualization for Dataset 1
st.write("## Data Preview - CQS.csv")
df_full = pd.read_csv(DEFAULT_CSV_PATH_CHATBOT)
st.write(df_full)  # Show the full DataFrame

# Display basic statistics for Dataset 1
st.write("## Data Statistics - CQS.csv")
st.write(df_full.describe(include='all'))  # Include all data types for descriptive stats

# Data visualization for Dataset 1
st.write("## Data Visualization - CQS.csv")

# Example: Countplot for 'Level'
if st.checkbox("Show Level Countplot - CQS.csv", key='level_countplot'):
    st.write("Countplot of Levels:")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Level', data=df_full, ax=ax, palette='viridis')  # Added palette for color
    st.pyplot(fig)

# Example: Time series plot for 'Timestamp'
if st.checkbox("Show Timestamp Time Series - CQS.csv", key='timestamp_timeseries'):
    st.write("Time Series of Events:")
    df_full['Timestamp'] = pd.to_datetime(df_full['Timestamp'])
    fig, ax = plt.subplots(figsize=(14, 7))
    df_full.groupby(df_full['Timestamp'].dt.date).size().plot(ax=ax, color='dodgerblue')  # Added color
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Events')
    ax.set_title('Number of Events per Day')
    st.pyplot(fig)

# Example: Ring chart for failed vs successful logins
if st.checkbox("Show Login Success vs Failure - CQS.csv", key='login_ring_chart'):
    st.write("Login Success vs Failure Ring Chart:")
    login_counts = df_full['Status'].value_counts()
    labels = login_counts.index
    sizes = login_counts.values
    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, 
                                      wedgeprops=dict(width=0.3), colors=sns.color_palette('pastel'))  # Added color palette
    ax.set_title('Failed vs Successful Logins')
    st.pyplot(fig)

# Add a separator line
st.write("---")

# Data Preview and Visualization for Dataset 2
st.write("## Data Preview - LOG.csv")
df_full2 = pd.read_csv(DEFAULT_CSV_PATH_DATASET2)
st.write(df_full2)  # Show the second DataFrame

# Display basic statistics for Dataset 2
st.write("## Data Statistics - LOG.csv")
st.write(df_full2.describe(include='all'))  # Include all data types for descriptive stats

# Data visualization for Dataset 2
st.write("## Data Visualization - LOG.csv")

# 1. Frequency of Status Codes
if st.checkbox("Show Status Code Frequency - LOG.csv", key='status_code_frequency2'):
    st.write("Frequency of Status Codes (LOG.csv):")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Status Code', data=df_full2, ax=ax, palette='cool')  # Added palette for color
    ax.set_title('Frequency of Status Codes')
    st.pyplot(fig)

# 2. Rate Limit Exceeded Analysis
if st.checkbox("Show Rate Limit Exceeded Count - LOG.csv", key='rate_limit_exceeded_count2'):
    st.write("Rate Limit Exceeded Analysis (LOG.csv):")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Rate Limit Exceeded', data=df_full2, ax=ax, palette='plasma')  # Added palette for color
    ax.set_title('Rate Limit Exceeded Analysis')
    st.pyplot(fig)

# 3. Proxy Jump Analysis with Line Chart
if st.checkbox("Show Proxy Jump Analysis - LOG.csv", key='proxy_jump_analysis2'):
    st.write("Proxy Jump Analysis (LOG.csv):")
    proxy_levels = df_full2['Proxy Jump'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
    proxy_levels_counts = proxy_levels.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    proxy_levels_counts.plot(kind='line', ax=ax, marker='o', color='darkorange')  # Changed to line chart and added color
    ax.set_title('Frequency of Proxy Levels Used Over Time')
    ax.set_xlabel('Proxy Level')
    ax.set_ylabel('Count')
    st.pyplot(fig)
