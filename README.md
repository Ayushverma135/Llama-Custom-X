# Llama-Custom-X - Chatbot Assistant 🤖


This repository contains a Streamlit-based web application named **Llama-Custom-X**, a Cyber Assistant designed to interact with CSV datasets and provide data-driven insights through a chatbot interface. The app leverages machine learning models to facilitate conversational queries and offers robust data visualization features.

## Features

- **Chatbot Interface**: Conversational interaction with the first dataset (`CQS.csv`). Ask questions about the data, and the AI will respond with insightful answers.
- **Data Preview and Visualization**: Visualize key aspects of the datasets (`CQS.csv` and `LOG.csv`) using count plots, time series, ring charts, and more.
- **Customizable and Expandable**: Easy to add more visualizations or modify existing ones based on your dataset requirements.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Ayushverma135/Llama-Custom-X.git
    cd Llama-Custom-X
    ```

2. **Install dependencies**:
    Make sure you have Python 3.8+ installed. Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the model**:
    Ensure that the `llama-2-7b-chat.Q4_0.gguf` model is available in your system. You can download it from GPT4All or Hugging Face.

## Usage

1. **Run the application**:
    ```bash
    streamlit run app.py
    ```

2. **Interact with the Cyber Assistant**:
   - Use the chatbot to query your data in the `CQS.csv` file. The assistant provides intelligent answers based on the CSV content.
   - Explore visualizations and statistics for both `CQS.csv` and `LOG.csv`.

## Dependencies

- `streamlit` - Web framework for creating the interactive app.
- `pandas` - Data manipulation and analysis library.
- `matplotlib` & `seaborn` - For data visualization.
- `langchain` - Framework for building language model-powered applications.
- `HuggingFaceEmbeddings` - To embed text for similarity search.
- `FAISS` - Facebook AI Similarity Search, used for fast similarity search.

## Installing `llama-2-7b-chat.Q4_0.gguf` Model from GPT4All and Hugging Face

### From GPT4All:

1. **Install GPT4All:**
   - If you haven't already, you can install GPT4All with the following command:
     ```bash
     pip install gpt4all
     ```

2. **Download the Model:**
   - Visit the [GPT4All website](https://gpt4all.io/index.html) to download the `llama-2-7b-chat.Q4_0.gguf` model. Ensure you download the `.gguf` version.
   - Alternatively, you can use a direct download link (if provided by GPT4All) to download the model.

3. **Load the Model:**
   - Once the model is downloaded, you can load it using GPT4All in Python:
     ```python
     from gpt4all import GPT4All

     model_path = "path/to/llama-2-7b-chat.Q4_0.gguf"
     gpt = GPT4All(model_path)
     ```

### From Hugging Face:

1. **Install Hugging Face Transformers:**
   - Install the `transformers` library if you don't have it:
     ```bash
     pip install transformers
     ```

2. **Install Hugging Face Hub (Optional):**
   - If you want to interact with models directly from the Hugging Face Hub:
     ```bash
     pip install huggingface_hub
     ```

3. **Download the Model:**
   - Use the Hugging Face model page to download `llama-2-7b-chat.Q4_0.gguf`. You may need to specify the correct variant if it’s available.
   - Alternatively, you can clone the model repository:
     ```bash
     git lfs install
     git clone https://huggingface.co/your-username/llama-2-7b-chat
     ```

4. **Load the Model:**
   - Load the model using the `transformers` library:
     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer

     model_name = "path/to/llama-2-7b-chat"
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     model = AutoModelForCausalLM.from_pretrained(model_name)
     ```

## Acknowledgements

- **Streamlit** for providing an easy-to-use framework for building interactive web apps.
- **LangChain** for making it easier to build language model applications.
- **Hugging Face** for their comprehensive library of machine learning models and embeddings.
- **FAISS** for enabling efficient similarity searches over large datasets.


## Notes:
- The specific path or way to load the model might vary depending on updates from GPT4All or Hugging Face, so always refer to the latest documentation.
- Ensure your environment has sufficient resources to handle large models like `llama-2-7b-chat`.

![app_v5](https://github.com/user-attachments/assets/dc552b45-bddc-4cb9-bf24-27a13877eb4d)
