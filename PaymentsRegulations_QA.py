import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import gradio as gr

# Parameters Setting 
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key_here"     

embedding_model_storage_directory = "./FAISS_storage/"

max_chunks_to_read = 6
score_threshold = 1.3
minimum_score_threshold = 1.2
# LLMs  Initialization
GPT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 200
TEMPERATURE = 0.2
llm = ChatOpenAI(model=GPT_MODEL, temperature = TEMPERATURE, max_tokens = MAX_TOKENS)

EMBEDDING_MODEL = "text-embedding-3-small"
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

parser = StrOutputParser()

# Templates Initialization
subject_template = ChatPromptTemplate.from_messages([
("system", 'Try to summarize the following task in the main subjects and provide this summary in no more than ten words.'),
("user", '{subject}')
])       

template = """Use the following corpus to answer the question at the end.
If the question is out of the scope of the corpus JUST SAY you don't know and don't try to make up an answer. 
If the question is within of the scope of the corpus prepare your answer and add a phrase saying Please also refet to:
as reference to the main Articles numbers and the Articles titles that were used to to make your answer. 
Use for your answer five sentences maximum and keep this answer as concise as possible.

{subject}
    
{corpus}

Question: {question}

Helpful Answer:
"""

# Initialize a Chain process for the retrieval of the Questions 
custom_reg_prompt = PromptTemplate.from_template(template)
    
reg_chain = (
    custom_reg_prompt
    | llm
    | StrOutputParser()
)

def Question_Subject(Question): 
    """ Find the subject of the Question """
    subject_prompt = subject_template.invoke({'subject': Question})
    subject_response = llm.invoke(subject_prompt)
    subject_str = parser.invoke(subject_response)
    
    return subject_str

def FAISS_Load(subject):
    """ Get the relevant chunks from the corpus vectorized data they are embedded and stored through the Ingestion process"""
    vector_store = LocalFileStore(embedding_model_storage_directory)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding_model, vector_store, namespace=embedding_model.model)

    # Load the FAISS index from disk
    if not Path(embedding_model_storage_directory).exists():
        raise ValueError(f"{embedding_model_storage_directory}  directory does not exist, please run ingest python file first")
    FAISS_retriever = FAISS.load_local(embedding_model_storage_directory, cached_embedder, allow_dangerous_deserialization=True)
    
    # Based on the Subject of the Question find the relevant documents 
    docs = FAISS_retriever.similarity_search_with_score(subject, k=max_chunks_to_read)
    return docs

def unique_doc_content(docs):
    """Check whether the content of the docs are the same or not and remain the unique ones"""
    unique_docs = []
    seen_page_content = set()
    for doc in docs:
        if doc.page_content not in seen_page_content:
            seen_page_content.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def Answer_Question(Question:str):
    """ Main processes for answering user's question """
    subject = Question_Subject(Question)   # getting the main subject of the Question by invoking the LLM
    retrieved_docs = FAISS_Load(subject)   # retieve documents from vectorized data, having 'subject' as input 

    # filtering retrieved_docs based of the similarity Score
    minimum_score = retrieved_docs[0][1] 
    retrieved_docs = [doc[0] for doc in retrieved_docs if ((doc[1] < score_threshold) & (doc[1]/minimum_score < minimum_score_threshold))]
    retrieved_docs = unique_doc_content(retrieved_docs)
    
    # Exiting when the Question is not relevant with corpus
    if len(retrieved_docs) <= 0: 
        response = "It seems that the question is not within the scope of the corpus"
        return response 
    
    retrieved_docs = format_docs(retrieved_docs) 
    
    # Get the 'question' and the relevant subject and based on the cospus (regulation) respond with an answer
    response = reg_chain.invoke({"corpus": retrieved_docs, "subject": subject, "question": Question})  
    return response

if __name__ == "__main__":
    demo = gr.Interface(fn=Answer_Question, inputs=["text"], outputs=["text"], title="Payments Regulation Q&A")
    demo.launch()