import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import TypedDict, List
from streamlit_chat import message
import streamlit.components.v1 as components
from markdown import markdown

# Function to get LLM
def get_llm():
    llm_type = os.getenv("LLM_TYPE", "ollama")
    if llm_type == "ollama":
        return ChatOllama(model="llama3.1:8b", temperature=0.7)
    else:
        return None

# Function to get embeddings
def get_embeddings():
    embedding_type = os.getenv("LLM_TYPE", "ollama")
    if embedding_type == "ollama":
        return OllamaEmbeddings(model="llama3.1:8b")
    else:
        return None

# Function to parse PDF files
def parse_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to parse TXT files
def parse_txt(file):
    return file.read().decode("utf-8")

# Function to summarize text
def summarize_text(text, llm):
    prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"
    summary = llm.invoke(input=prompt)
    return summary.content

# Define the agent state and functions
class AgentState(TypedDict):
    question: str
    grades: List[str]
    llm_output: str
    documents: List[str]

def retrieve_docs(state: AgentState, retriever):
    question = state["question"]
    documents = retriever.invoke(input=question)
    state["documents"] = [doc.page_content for doc in documents]
    st.write("Retrieved Documents:", state["documents"])
    return state

class GradeDocuments(BaseModel):
    """Boolean values to check for relevance of retrieved documents."""
    score: str = Field(description="Documents are relevant to the question, 'Yes' or 'No'")

def document_grader(state: AgentState):
    docs = state["documents"]
    question = state["question"]
    system = """You are a grader assessing the relevance of a retrieved document to a user question.
Your task is to determine if the content of the document is directly relevant to answering the user's question.

Consider a document relevant if it:
1. Contains keywords related to the question.
2. Addresses the topic or context of the question.
3. Provides information that could help in answering the question.

Examples:
- Question: "What is AI?" Document: "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines..." -> Yes
- Question: "How does a neural network work?" Document: "Neural networks are a series of algorithms that attempt to recognize underlying relationships in a set of data..." -> Yes
- Question: "Explain reinforcement learning." Document: "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions..." -> Yes

Please give a binary score 'Yes' or 'No' to indicate whether the document is relevant to the question. If the document is relevant, respond with 'Yes'. If the document is not relevant, respond with 'No'."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeDocuments)
    grader_llm = grade_prompt | structured_llm
    scores = []
    for doc in docs:
        result = grader_llm.invoke({"document": doc, "question": question})
        scores.append(result.score)
    state["grades"] = scores
    st.write("Document Grades:", scores)
    return state

def gen_router(state: AgentState):
    grades = state["grades"]
    if any(grade.lower() == "yes" for grade in grades):
        return "generate"
    else:
        return "rewrite_query"

def rewriter(state: AgentState):
    question = state["question"]
    system = """You are a question re-writer that converts an input question to a better version that is optimized
        for retrieval from a specific set of documents. Ensure that the rewritten question is directly relevant to the context of the documents and avoid introducing any new information or hallucinations. Focus on understanding the underlying semantic intent and making the question as clear and precise as possible. Do not speculate or add any information not present in the original question.
        
        Examples:
        Original: What is the latest research in AI?
        Rewritten: Can you summarize the most recent advancements in artificial intelligence as per the provided documents?

        Original: Explain machine learning.
        Rewritten: How is machine learning defined and explained in the provided documents?

        Original: What is the capital of France?
        Rewritten: [This question cannot be answered as it is not relevant to the provided documents.]
        """
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )
    llm = get_llm()
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    output = question_rewriter.invoke({"question": question})
    state["question"] = output
    st.write("Rewritten Question:", output)
    return state

def generate_answer(state: AgentState):
    llm = get_llm()
    question = state["question"]
    context = state["documents"]
    template = """Answer the question based on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})
    state["llm_output"] = result
    st.write("Generated Answer:", result)
    return state

workflow = StateGraph(AgentState)

workflow.add_node("retrieve_docs", lambda state: retrieve_docs(state, retriever))
workflow.add_node("rewrite_query", rewriter)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("document_grader", document_grader)

workflow.add_edge("retrieve_docs", "document_grader")
workflow.add_conditional_edges(
    "document_grader",
    gen_router,
    {
        "generate": "generate_answer",
        "rewrite_query": "rewrite_query",
    },
)
workflow.add_edge("rewrite_query", "retrieve_docs")
workflow.add_edge("generate_answer", END)

workflow.set_entry_point("retrieve_docs")

app = workflow.compile()

# Initialize Streamlit app
st.set_page_config(
    page_title="DocuMentor AI",
    page_icon="🤖"
)


st.markdown('<h1>Docu<span style="color:#5B99C2">Mentor</span> AI 🤖</h1>', unsafe_allow_html=True)
# Initialize chat history
if "history" not in st.session_state:
    st.session_state["history"] = []
if "upload" not in st.session_state:
    st.session_state["upload"] = []
if "docs" not in st.session_state:
    st.session_state["docs"] = []
if "summaries" not in st.session_state:
    st.session_state["summaries"] = []
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

history = st.session_state["history"]

llm = get_llm()
embedding_function = get_embeddings()
# File upload
uploaded_files = st.file_uploader("Upload your documents (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if len(uploaded_files) != len(st.session_state["upload"]):
    docs = []
    summaries = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file in st.session_state["upload"]:
            continue
        if uploaded_file.type == "application/pdf":
            text = parse_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = parse_txt(uploaded_file)
        st.session_state["upload"].append(uploaded_file)
        summary = summarize_text(text, llm)
        summaries.append(summary)
        
        
        doc = Document(page_content=text)
        if doc not in st.session_state["docs"]:
            st.session_state["docs"].append(doc)
    # Split documents using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = splitter.split_documents(st.session_state["docs"])
    

    # Initialize Chroma database with the split documents and embeddings
    
    db = Chroma.from_documents(split_docs, embedding_function)
    retriever = db.as_retriever()
    st.session_state["retriever"] = retriever
    st.session_state["summaries"].append(summaries)
    st.success("Documents have been successfully parsed, summarized, split, and stored in Chroma.")
    
if st.session_state["summaries"]:        
    st.write("Summaries:")
    for uploaded_file, summary in zip(uploaded_files, st.session_state["summaries"]):
        with st.expander(f"Summary of {uploaded_file.name}"):
            st.write(summary[0])
    
      
    
if st.session_state["retriever"]:
        retriever = st.session_state["retriever"]

        # Scrollable chat history container
        st.header("Chat History")
        chat_history_html = '''
        <style>
            .chat-container {
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ccc;
            }
            .chat-container .message {
                color: white;
                background-color: rgba(30, 30, 30, 0.7);
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                white-space: pre-wrap;
                font-family: 'Arial', sans-serif;  /* Use Streamlit's default font */
                font-size: 16px;  /* Adjust font size as needed */
                backdrop-filter: blur(10px);  /* Apply blur for glass effect */
            }
            .chat-container .user {
                background-color: rgba(16, 44, 87,0.7);  /* Yellow glass effect for user */
            }
            .chat-container .assistant {
                background-color: rgba(117, 134, 148,0.4);  /* Green glass effect for assistant */
            }
            .chat-container .user::before {
                content: "User: ";
                font-weight: bold;
                color: rgb(54, 194, 206);
            }
            .chat-container .assistant::before {
                content: "Chatbot: ";
                font-weight: bold;
                color: rgb(54, 194, 206);
            }
        </style>
        <div class="chat-container">
        '''
        for message_dict in history:
            user_class = "user" if message_dict["type"] == "user" else "assistant"
            html_content = markdown(message_dict["content"])
            chat_history_html += f'<div class="message {user_class}">{html_content}</div>'
        chat_history_html += '</div>'
        components.html(chat_history_html, height=400)


        # Chatbot interface
        st.header("Chatbot Q&A")
        user_question = st.text_input("Ask a question about the uploaded documents:")
        if st.button("Get Answer"):
            state = app.invoke({"question": user_question})
            st.write("Answer:", state["llm_output"])
            history.append({"type": "user", "content": user_question})
            history.append({"type": "assistant", "content": state["llm_output"]})
