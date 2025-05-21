#!/usr/bin/env python
import pypdf
import csv
import pickle
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import argparse
from torch.utils.data import DataLoader,default_collate
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm import tqdm
from timeit import timeit
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DistanceStrategy

from utils import *

#self.retrieval = AutoModelForSentenceEmbeddings(**retrieval_config)
#self.tokenizer_reranker = AutoTokenizer.from_pretrained(**generator_config)
    
class RAG_Chatbot:

    def __init__(self,
                 retrieval_config = retrieval_config,
                 reranker_config = reranker_config,
                 generator_config = generator_config,
                 docs_path = r'./src/Documents',
                 output_path = r'./src/Output',
                 embeddings_path =  r'./src/Embeddings',
                 loader_cls = PyPDFLoader,
                 chunk_size = 512,
                 separators = ['\n\n','\n',' ',''],
                 batch_size = 200,
                 collate_fn = default_collate,
                 top_k = 10,
                 generation_config = generation_config 
    ):
        self.retrieval = AutoModelForSentenceEmbeddings(**retrieval_config)
        self.generator = AutoModelForCausalLM.from_pretrained(**generator_config)
        self.tokenizer_retrieval = AutoTokenizer.from_pretrained(**retrieval_config)
        self.tokenizer_generator = AutoTokenizer.from_pretrained(**generator_config)
        self.docs_path = docs_path
        self.output_path = output_path
        self.embeddings_path = embeddings_path
        self.loader_cls = loader_cls
        self.chunk_size = chunk_size
        self.separators = separators
        self.device = torch.device('cuda:0')
        self.num_workers = torch.cuda.device_count()
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.top_k = top_k
        self.generation_config = generation_config 

    def load_documents(self, path = None):
        if path == None:
            path = self.docs_path
        # Load
        loader = DirectoryLoader(
            path = path,
            loader_cls = self.loader_cls
        )
        documents = loader.load()
        # Preprocess
        documents = update_docs_metadata(documents)
        documents = keep_top_content(documents,fraction = 0.4)
        # Split
        chunk_size = self.chunk_size
        chunk_overlap = int(chunk_size*0.1)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators = self.separators
        )
        documents = text_splitter.split_documents(documents)
        return documents

    def get_embeddings(self,texts,batch_size = None):
        if batch_size == None:
            batch_size = len(texts)
        batches = get_batches(texts, batch_size)
        tokenizer = self.tokenizer_retrieval
        model = self.retrieval
        device = self.device
        embeddings = torch.tensor([],device=device)
        with torch.no_grad():
            for batch in tqdm(batches):
                encoded_input = tokenizer(batch, padding = True, truncation = True, return_tensors = 'pt').to(device) 
                embeddings = torch.cat([embeddings,model(**encoded_input)])
        return embeddings

    def retrieve_top_k(self,embeddings_1,embeddings_2,query,documents):
        k = self.top_k
        scores = torch.mm(embeddings_1,embeddings_2.t())
        scores,indexes = scores.sort(dim=1,descending = True)
        scores = scores[:,:k]
        indexes = indexes[:,:k]
        return build_retrieval_output(indexes,scores,query,documents)

    def generate_answer(self,query):
        # Query as a list
        if type(query) == str:
            query = [query]
        # Initialize generation objects
        device = self.device
        tokenizer = self.tokenizer_generator
        tokenizer.pad_token = tokenizer.eos_token
        generation_config = self.generation_config
        model = self.generator
        # Retrieve top similar documents
        batch_size = self.batch_size
        embeddings_path = self.embeddings_path 
        if os.path.exists(embeddings_path) == False:
            documents = self.load_documents()
            document_contents = [document.page_content for document in documents]
            document_embeddings = self.get_embeddings(document_contents,batch_size)
            os.makedirs(embeddings_path)
            with open(embeddings_path+'/documents.pickle','wb') as file:
                pickle.dump(documents,file)
            torch.save(document_embeddings,embeddings_path +'/embeddings.pt')
        else: 
            with open(embeddings_path+'/documents.pickle','rb') as file:
                documents = pickle.load(file) 
            document_embeddings = torch.load(embeddings_path +'/embeddings.pt')
        query_embeddings = self.get_embeddings(query)
        query_top_documents = self.retrieve_top_k(query_embeddings,document_embeddings,query,documents)
        # Generate answers
        text_input = prepare_generation_input(query_top_documents)
        encoded_input = tokenizer(text_input, padding = True, truncation = True, return_tensors = 'pt').to(device)
        with torch.no_grad():
            encoded_output = model.generate(**encoded_input,generation_config = generation_config)
        # Post-process output
        answers_text_input = tokenizer.batch_decode(encoded_output,skip_special_tokens = True)
        text_input_cleaned = tokenizer.batch_decode(encoded_input['input_ids'],skip_special_tokens = True)
        answers = [item[0][len(item[1]):].replace('\n\n','\n').strip('\n')\
                   for item in zip(answers_text_input,text_input_cleaned)]
        output = [{'Query':qa[0],'Result':qa[1]} for qa in zip(query,answers)]
        return output,query_top_documents

    
if __name__ == '__main__':

   # parser = ArgumentParser()
  #  parser.add_argument('-q','--query', default = 'Test')
   # args = parser.parse_args()

    # for query in queries:

    rag = RAG_Chatbot()
    output_path = rag.output_path
    batch_size = 10

    query = ['What attention is used in Llama 3?',
             'When was Claude 3 released?',
             'Give me an overview of Gemma 2.',
             'What are GPT-4o capabilities?',
             'What is Gemini 1.5 architecture?',
             'How is Claude 3 pre-trained?',
             'What data is used for pre-training GPT-4o?',
             'How is Gemini 1.5 post-trained?',
             'How is Gemma 2 evaluated?',
             'What is Llama 3 performance on benchmarks?']

    #query = 'What attention is used in Llama 3?'
    #query = query[0:5]
    output,retrieval_output = rag.generate_answer(query)
    complete_output = build_complete_output(output,retrieval_output)
    output_df = pd.DataFrame.from_dict(output)
    retrieval_output_df = pd.DataFrame.from_dict(retrieval_output)
    #complete_output_df = pd.Series(complete_output)
    if os.path.exists(output_path) == False:
        os.makedirs(output_path)
    output_df.to_csv(output_path+'/output.csv',index = False)
    retrieval_output_df.to_csv(output_path+'/retrieval_output.csv',index = False)
    #complete_output_df.to_csv(output_path+'/complete_output.csv',index = False)
    with open(output_path+'/complete_output.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(complete_output)
