# Loading Libraries 
import os
import glob

import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings

# Setting Parameters
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key_here"    

# Data source file
SourceFileName = './source/PSR_2023_6_Source.txt'
ChunksFileName = './intermediate_data/Regulation_Chunks.csv'
embedding_model_storage_directory = "./FAISS_storage/"

# Initialize the Splitter
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200

text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP, add_start_index=True)

# Initialize the embedding model (this example uses OpenAI's embeddings)
EMBEDDING_MODEL = "text-embedding-3-small"

embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

def Load_Source_Data():
    """ Loading Source Data """
    try:
        loader = TextLoader(SourceFileName)
        docs = loader.load()
        return docs
    except FileNotFoundError:
        print("Error: The file was not found.")
    except PermissionError:
        print("Error: You don't have permission to read the file.")
    except IsADirectoryError:
        print("Error: Expected a file, but found a directory.")
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_articles(text):
    """ This function splits the regulation text into chunks of text but also keeps additional information 
    like the Article's Number and Title so as this information to be avavailable as reference in the chunks text"""

    # Initialization of a regulation articles dictionary to keep the necessary information
    reg_atricles = {
        'Article_Topic': [],
        'Article_Prefix': [],
        'Article_Number': [], 
        'Article_Title': [],
        'Chunk_Number': [],
        'Chunk': []
        }
    chunk_num = 0

    # Split the text in topics based on the ^^^ marker
    topics = text.split('^^^')
    for topic in topics: 
    
        lines = topic.split('\n')
        topic_text = lines[0].strip()  # the first phrase of each topic regards description of the regulation TITLES or CHAPTERS
        
        # Split the text in articles based on the @@@ marker
        articles = topic.split('@@@')
    
        # Remove empty strings and strip whitespace
        articles = [article.strip() for article in articles if article.strip()]
        
        # Prepare the final list of articles
        for article in articles:
            # Ensure the chunk includes the marker to maintain clarity on what the chunk represents
            lines = article.split('\n')
        
            # Extract the article number and title from the first line
            firstline  = lines[0].strip()  # the first sentence of each article is of "Article x" form
            number = firstline.split()[1].strip()   # Assumes format "Article X" to get the number of the article
            if "Article" in firstline:
                """ filtering Atciles text vs. Titles and Subtitles of the document """
                sedcondline = lines[1].strip()        
                content = '\n'.join(lines[2:]).strip()  # Extract the article content from the third line and the nect
            
                chunks = text_splitter.split_text(content)    # Split the Article's content to chunks
                for chunk in chunks: 
                    """ for every chunk keep necessary data in the form of a dictionary """                     
                    reg_atricles['Article_Topic'].append(topic_text)
                    reg_atricles['Article_Prefix'].append(firstline),
                    reg_atricles['Article_Number'].append(number), 
                    reg_atricles['Article_Title'].append(sedcondline),            # the second sentence of each article is the Title of the article    
                    reg_atricles['Chunk_Number'].append(chunk_num),
                    #the chunks' text is refixed with Article Number and Title to provide reference     
                    reg_atricles['Chunk'].append(f"Topic: {topic_text}, Article: {number}, Title: {sedcondline}\n{chunk}")  
                                
                    chunk_num += 1    
    
    return reg_atricles
    
def Save_to_csv():
    try:
        articles_df.to_csv(ChunksFileName)
        print(f"Data successfully saved to {ChunksFileName}")
    except IOError as e:
        print(f"Failed to write to {ChunksFileName}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def Embed_and_Save_data(chunks):

    # Creation of Documents from Chunks of the Regulation Articles DataFrame
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Initialization of FAISS 
    store = LocalFileStore(embedding_model_storage_directory)

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embedding_model, store, namespace=embedding_model.model)

    # Embedding through FAISS 
    print("Embedding process started. It will take a few a minutes..... ")
    vector_store = FAISS.from_documents(docs, cached_embedder)

    # Delete all files in the embedding storage directory
    files = glob.glob(f"{embedding_model_storage_directory}*")
    for file in files:
        os.remove(file)
    # Saving embedding results
    vector_store.save_local(embedding_model_storage_directory)
    print(f"Embeded Data successfully saved to {embedding_model_storage_directory} directory")

if __name__ == "__main__":
    """ Main Process """

    # Loading Source Data
    documents = Load_Source_Data()

    # Call the function to split into articles
    regulation_articles = get_articles(documents[0].page_content)

    articles_df = pd.DataFrame(regulation_articles)
    articles_df.head()
    Save_to_csv()

    Embed_and_Save_data(regulation_articles['Chunk'])