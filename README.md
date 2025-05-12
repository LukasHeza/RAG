# RAG on 2024 LLMs arXiv papers using LangChain and Llama-3.1
## Overview
Retrieval Augmented Generation (RAG) to answer questions about LLMs released in 2024 based on arXiv research papers; Python, LangChain and Transformers library.
- Llama-3.1-8B-Instruct
- RAG built as a model (class)
- Batch input
- Output contains generated answers and retrieved documents

### Retrieval Augmented Generation    
- Technique that improves LLM accuracy for text generation by extending an input question with documents containing the answer.   
- This way LLM receives additional domain-specific or updated data that helps to overcome limitations of its static training dataset. 
- Document retrieval (matching a question against the entire set of documents) is used to find the documents containing the answer.   
- The documentation can come from inner company or online sources.
     
**Advantages**:   
- LLM does not need to be further trained, meaning:
  - There is no need to have labeled data.
  - Saves time and computational resources required for training.
- Dynamic – can accesses data that are updated in real time.
- Flexible - can be easily used on another domain by replacing the source of documents.     

## System Requirements
    GPU T4 (15GB RAM) x2
    Runtime: 5m 53s 

Libraries:

    transformers==4.46.1 
    accelerate==0.26.0 
    bitsandbytes==0.44.1 
    pandas==2.2.3 
    pypdf==5.1.0 
    langchain==0.3.14 
    langchain-community==0.3.14 
    langchain-huggingface==0.1.2 
    faiss-cpu==1.9.0.post1 
    --extra-index-url https://download.pytorch.org/whl/cu124 torch
    
## How to use
LLMs arXiv release papers can be found in the `./src/Documents`. The `rag.py` script is in the `./src/Model`.  
```
.
├── requirements.txt
└── src
    ├── Documents
    │   ├── 2403.05530v4.pdf
    │   ├── 2407.21783v2.pdf
    │   ├── 2408.00118v3.pdf
    │   ├── 2410.21276v1.pdf
    │   └── Model_Card_Claude_3.pdf
    ├── Model
    │   ├── __pycache__
    │   │   └── utils.cpython-311.pyc
    │   ├── rag.py
    │   └── utils.py
    └── Output
        ├── complete_output.csv
        ├── output.csv
        └── retrieval_output.csv
```
To run inference use the [Kaggle notebook](https://www.kaggle.com/code/lukasheza/rag-inference).
```
$ git clone https://github.com/LukasHeza/RAG.git
 # to copy directory to your local machine
$ cd RAG
 # to change parent directory
$ python -m pip install -r requirements.txt
 # to install the libraries
$ ./src/Model/rag.py
 # to execute rag python script
```

## RAG steps  
1. Loads documents. 
2. Splits documents into smaller chunks - passages.  
3. Embeds passages and query.     
4. Stores embedded passages in a vector database.     
5. Retrieves k most relevant passages to the query.   
6. Builds input/prompt by combining retrieved texts, query and instructions.   
7. Feeds the input to the LLM to generate answer/s.   
8. Post processes output.   
