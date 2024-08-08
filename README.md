# DocuMentor AI 🤖

DocuMentor AI is an intelligent document management and chatbot application designed to efficiently parse, summarize, and retrieve information from PDF and TXT files. It leverages advanced language models and Agentic Retrieval-Augmented Generation (RAG) to provide context-aware answers and enhanced user interactions.

## Features

- **Document Parsing and Summarization**: Upload PDF and TXT files, which are automatically parsed and summarized.
- **Efficient Information Retrieval**: Uses vector databases to store and retrieve relevant document chunks.
- **AI-Powered Question Answering**: Provides accurate answers to user queries based on the uploaded documents.
- **Automatic Document Grading**: Assesses the relevance of documents to user questions.
- **Question Rewriting**: Optimizes user questions for better retrieval and accuracy.
- **Agentic RAG**: Employs Agentic RAG to intelligently combine retrieval and generation for enhanced document understanding and interaction.

## Technologies Used

- **Python**: Core programming language.
- **PyPDF2**: For parsing PDF files.
- **LangChain**: For integrating language models and handling document retrieval.
- **Chroma**: For vector storage and retrieval.
- **Streamlit**: For creating the web interface.
- **Markdown**: For formatting text in the chat interface.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/AfnanHussain10/DocuMentor-AI.git
    cd DocuMentor-AI
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file and add the following:
      ```sh
      LLM_TYPE=ollama
      ```

## Usage

1. **Run the Streamlit app**:
    ```sh
    streamlit run documentor.py
    ```

2. **Upload documents**:
    - Upload your PDF and TXT files through the provided interface.
    - The documents will be parsed, summarized, and stored.

3. **Interact with the chatbot**:
    - Ask questions related to the uploaded documents.
    - View the chat history and document summaries.
