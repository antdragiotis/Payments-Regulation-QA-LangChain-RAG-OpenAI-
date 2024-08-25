## QA EU Payments Regulations (OpenAI, LangChain)
This application utilizes **OpenAI** and **LangChain** under a **Retrieval Augmented Generation (RAG)** to retrieve information from the European **Payment Services Regulation (PSR)**. 
**RAG** is an approach that combines the strengths of retrieval-based models and generation-based modelsto improve  accuracy and relevance of generated text by leveraging external information sources. 
Initially, the **PSR** text is embedded and stored for efficient retrieval though the functionality of **FAISS** library. A secondary process then uses this embedded data, combined with **OpenAI** services, to identify the most relevant information in response to user queries, and formulate a final answer in natural language.

### Purpose 
European regulations governing banks, payments, and financial services are both critical and highly complex. They encompass a vast array of extensive, interdependent texts that demand strict compliance. These regulations are frequently updated, making it challenging for institutions to stay current. The broad scope of these laws, requires financial institutions to constantly manage and interpret a massive volume of regulatory material, which is essential for maintaining legal compliance and avoiding significant penalties. The complexity and interconnectivity of these texts underscore the need for advanced tools to efficiently manage regulatory compliance.

### Process Overview 
The ingestion and quering of the CRR contents follows the below steps: 

![Process Overview](https://github.com/antdragiotis/Payments-Regulation-QA-LangChain-RAG-OpenAI-/blob/main/QA_PSR_processes.PNG)

### Features
- **Source Data**: The application uses as source data file the **PSR_2023_6_Source.txt** which is a text file version of the European Payment Services Regulation (PSR). PSR along with the third Payment Services Directive (PSD3) are a new set of legislative proposals from the European Commission that bring changes to the foundational framework of the European payments market. The original PSR text is available at EUR-Lex (https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52023PC0367). For this  application, only the regulation articles have been retained. The introductory sections have been excluded, resulting in a focused text file to serve as the application's source data.
- **Ingestion**: The 'Ingestion' process is performed by the *PaymentsRegulations_Ingestion.py* Python code file. This process reads the Source Data and splits the PSR text into chunks, assigning also to each chunk information about the relevant article number, article title and the regulation topic (according to the TITLES of the text). It saves this information into the *Regulation_Chunks.csv* file in the *intermediate_data* directory. Next to this, the process uses **FAISS** library to vectorize chunks text and store the results to the *FAISS_storage* directory. 
- The *PaymentsRegulations_QA.py* Python script executes three key processes to gather user questions and provide responses:
  - **Question Summarization**: Utilizes an OpenAI model to generate a concise, maximum ten-word summary of the end user's question. While an alternative approach involves directly matching the user’s question with vectorized data, summarizing the question first and then searching the vectorized data for key points is preferred. This way enables the application to better capture the essence of a wider range of free-form questions.  
  - **Retrieval of Embedded Data** the question summary is compared with the embedded (vectorized) data to find the most relevant chunks of the regulation text.
  - **Text Generation** The selected chunks (corpus) along with the end user question and relevant instructions formulate a prompt for the OpenAI model to respond. The model is explicitly instructed to base its response on the selected corpus chunks and to refrain from generating information outside the scope of the regulation or when the answer is unknown. Additionally, the model provides references to the specific articles of the regulation that were used in formulating its response.
- **Sample responses**: the file has the following columns: The *Sample_Results.csv* file, located in the *results* directory, contains sample responses generated by the application for various questions. The inclusion of certain questions outside the scope of the regulation is intentional. This  is  to demonstrate that the application's responses are firmly grounded in the provided corpus and not merely reflective of the LLM's training process. The file includes the following columns:
  - **Question**: the question that a hypothetical user has made to the application
  - **Question_Subject**: the question's summary (key points) as they have been addressed by the OpenAI model
  - **Articles_Chunks**: how many text chunks has the application retrieved from the vectorized data in responce to the question
  - **Answer**: the answer of the question as is has to be presented to the user   

- ### How to run the app:
- clone the repository: https://github.com/antdragiotis/Payments-Regulation-QA-LangChain-RAG-OpenAI-
- change current directory to cloned repository
- pip install -r requirements.txt
- it is assumed that your system has been configured with the OPENAI_API_KEY and LANGCHAIN_API_KEY, otherwise you need to add the following statements to the python code files:
  - import os
  - os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
  - os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key_here"     
- run the two Python files as the processes described above: 
  - PaymentsRegulations_INGEST.py
  - PaymentsRegulations_QA.py
 
You get pre-generated sample results the results in the *results/Sample_Results.csv* file, but the second Python file initiates a *gradio* UI service where you can insert your question and receive a relevant answer, as the screen below: 

![UI](https://github.com/antdragiotis/Payments-Regulation-QA-LangChain-RAG-OpenAI-/blob/main/QA_PSR_UI.PNG)

### Project Structure
- *.py: main application code
- source_data: directory with the source data file
- intermediate_data: directory with intermediate data file that facilitate review of the ingestion process
- results: directory with a sample  output file
- README.md: project documentation
