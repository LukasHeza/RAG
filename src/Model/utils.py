import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer,BitsAndBytesConfig,AutoConfig,GenerationConfig
import re
import os

access_token = os.environ['HF_token']
quantization_config = BitsAndBytesConfig(load_in_8bit = True)
generation_config = GenerationConfig(
    max_new_tokens = 200,
    do_sample = False,
    num_beams = 1
)

#'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
#Alibaba-NLP/gte-base-en-v1.5#
# thenlper/gte-base
#'google/gemma-2-2b-it'#
#'meta-llama/Llama-3.1-8B-Instruct'

retrieval_config = {
    'pretrained_model_name_or_path':'thenlper/gte-base',
    'quantization_config':quantization_config,
    'device_map':'auto'
}

retrieval_config = {
    'pretrained_model_name_or_path':'thenlper/gte-base',
    'device_map':'auto'
}

reranker_config = {
    'pretrained_model_name_or_path':'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    'quantization_config':quantization_config,
    'device_map':'auto'
}

generator_config = {
    'pretrained_model_name_or_path':'meta-llama/Llama-3.1-8B-Instruct',
    'quantization_config':quantization_config,
    'device_map':'auto',
    'token':access_token
}

patterns = {
    'title':r'([A-Z].*\n?[a-z].*\n?[a-z].*)\n[A-Z]',
    'heading_I':'[0-9]\.? +[A-Z].{0,50}[A-Za-z]{3,}',
    'heading_II':'[0-9]\.[0-9]\.? +[A-Z].{0,60}[A-Za-z]{3,}',
    'heading_III':'[0-9]\.[0-9]\.[0-9]\.? +[A-Z].{0,60}[A-Za-z]{3,}', 
    'separator':'\.'
}
separators = ['\n'+pattern+'\n' if pattern != list(patterns.values())[-1] else 
              pattern+' '+'|'+pattern+'\n' for pattern in list(patterns.values())]+['\n',' ','']

separators = ['\n\n','\n',' ','']

class AutoModelForSentenceEmbeddings(nn.Module):
    def __init__(self,**config):
        super().__init__()
        self.model = AutoModel.from_pretrained(**config)
        self.config = AutoConfig.from_pretrained(config['pretrained_model_name_or_path'])

    def forward(self,**kwargs):
        token_embeddings = self.model(**kwargs)[0]
        if 'Causal' in self.config.architectures[0]:
            pooling = self.last_token_pooling
        else:
            pooling = self.mean_pooling
        return pooling(token_embeddings,kwargs['attention_mask'])    

    def last_token_pooling(self,token_embeddings,attention_mask):
        last_token_idxs = attention_mask.sum(1).type(torch.int64) - 1
        sequence_idxs = torch.arange(token_embeddings.size()[0])
        return F.normalize(token_embeddings[sequence_idxs,last_token_idxs],p = 2,dim = 1)

    def mean_pooling(self,token_embeddings,attention_mask):
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sentence_embeddings = (token_embeddings * attention_mask_expanded).sum(1)/attention_mask_expanded.sum(1)
        return F.normalize(sentence_embeddings,p = 2,dim = 1)
        
        
def keep_top_content(documents,fraction = 0.4):
    top_contents = []
    counts_per_source = count_documents_per_source(documents)
    for document in documents:
        document_source = document.metadata['source']
        page_threshold = int(counts_per_source[document_source] * fraction)
        document_page = document.metadata['page']
        if document_page <= page_threshold:
            top_contents.append(document)
    return top_contents

def count_documents_per_source(documents):
    counts_per_source = {}
    count = 1
    document_source0 = documents[0].metadata['source']
    for i in range(1,len(documents)):
        document = documents[i]
        document_source = document.metadata['source']
        if document_source == document_source0:
            count+=1
        else:
            counts_per_source[document_source0] = count
            document_source0 = document_source
            count = 1
    counts_per_source[document_source] = count
    return counts_per_source
    
def update_docs_metadata(documents):
    pattern = r'([A-Z].*\n?[a-z].*\n?[a-z].*)\n[A-Z]'
    for document in documents:
        if document.metadata['page'] == 0:
            title = re.findall(pattern,document.page_content)[0]
        document.metadata['title'] = title
    return documents
    
def cleanse_doc_content(documents):
    for document in documents:
        document.page_content = document.page_content.replace('\n',' ')
    return documents

def adjust_separator(documents,separator = patterns['separator']):
    separator = separator.replace('\\','')
    for document in documents:
        document.page_content = document.page_content.lstrip(separator+' ').lstrip(separator+'\n ') + separator
    return documents

def merge_headings_content(headings_,headings,content):
    n_heading_ = [heading_ for heading_ in headings_ if heading_!='']
    if len(n_heading_) == 0:
        headings_output = ': '.join([heading for heading in headings if heading!=''])
        if headings_output != '':
            content = headings_output+'\n'+content
    else:
        for idx,heading_ in enumerate(headings_[1:]):
            idx+=1
            if heading_ != '':
                headings_output = ': '.join(headings[:idx+1])
                content = content.replace(heading_,headings_output)
    return content
    
def add_doc_headings_title(documents, patterns = patterns):
    heading_keys = list(patterns.keys())[1:-1]
    separator = patterns['separator']
    document = documents[0]
    content = document.page_content
    source = document.metadata['source']
    title = re.findall(patterns['title'],content)[0] 
    headings = ['' for i in range(len(heading_keys))]
    for document in documents:
        content = document.page_content
        source_ = document.metadata['source']
        if source_ != source:
            title = re.findall(patterns['title'],content)[0]
            headings = ['' for heading in headings]
            source = source_
        document.metadata['title'] = title
        headings_ = []
        heading_ = ''.join(re.findall(f'^({patterns[heading_keys[0]]})\n',content)) 
        if heading_ == '':
            if ''.join(re.findall(patterns[heading_keys[0]],content)) == content.rstrip(separator):
                heading_ = content
        if heading_ != '':
            headings[0] = heading_
        headings_.append(heading_)
        #document.metadata[heading_keys[0]] = headings[0]
        for idx,heading_key in enumerate(heading_keys[1:]):
            idx+=1
            heading__ = ''.join(re.findall(f'^({patterns[heading_key]})\n',content))
            if heading__ == '':
                heading__ = ''.join(re.findall(f'\n({patterns[heading_key]})\n',content))
            if heading__ != '':
                headings[idx] = heading__
            else:
                if heading_ != '':
                    headings[idx] = ''
            heading_ = heading__
            headings_.append(heading__)
            #document.metadata[heading_key] = headings[idx]
        content = merge_headings_content(headings_,headings,content)
        headings_output = ': '.join([heading for heading in headings if heading!=''])
        document.metadata['headings'] = headings_output
        document.page_content = content
    return documents

def build_dataset(documents):
    return [{'text':document.page_content,'title':document.metadata['title'],
             'source':document.metadata['source']} for document in documents]

def get_batches(texts, batch_size):
    for i in range(0,len(texts),batch_size):
        yield texts[i:i+batch_size]

def build_retrieval_output(indexes,scores,queries,documents):
    retrieval_output = []
    n_queries = indexes.size()[0]
    k_documents = indexes.size()[1]
    for i in range(n_queries):
        query = queries[i]
        for j in range(k_documents):
            output = {}
            idx = indexes[i,j].item()
            document = documents[idx]
            score = scores[i,j].item()
            output['query_idx'] = i
            output['doc_idx'] = idx
            output['doc_score'] = score
            output['query'] = query
            output['doc_content'] = document.page_content
            output['doc_title'] = document.metadata['title']
            output['doc_source'] = document.metadata['source']
            output['doc_page'] = document.metadata['page'] + 1
            
            retrieval_output.append(output)
    return retrieval_output

def prepare_generation_input(top_documents):
    text_input = []
    context = ''
    for i in range(len(top_documents)-1):
        top_document = top_documents[i]
        top_document2 = top_documents[i+1]
        query_idx = top_document['query_idx']
        query_idx2 = top_document2['query_idx']
        doc_title = top_document['doc_title']
        doc_content = top_document['doc_content']
        question = top_document['query'] 
        if query_idx == query_idx2:
           # context += doc_title+'\n'+doc_content+'\n\n'
             context += doc_content+'\n'
        else:
            #context += doc_title+'\n'+doc_content
            context += doc_content
#            text = f'''Answer the question based on the context.
#Question:{question}
#Context:{context}
#Answer:'''
            text = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Answer the question based on the context.
Context:{context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
            text_input.append(text)
            context = ''
    doc_title = top_document2['doc_title']
    doc_content = top_document2['doc_content']
    #context += doc_title+'\n'+doc_content+'\n\n'
    context += doc_content
#    text = f'''
#Answer the question based on the context.
#Question:{question}
#Context:{context}
#Answer:
#'''
    text = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Answer the question based on the context.
Context:{context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    text_input.append(text)
    return text_input

def build_complete_output(output,retrieval_output):
    complete_outputs = []
    complete_output = ''
    for i in range(len(output)):
        for item in output[i].items():
            complete_output += f'{item[0]}: {item[1]}\n'
        complete_output += 'Documents:\n'
        for j in range(len(retrieval_output)):
            if retrieval_output[j]['query_idx'] == i:
                for idx,item in enumerate(retrieval_output[j].items()):
                    if idx > 3:
                        complete_output += f'{item[0]}: {item[1]}\n'
                complete_output += '\n'
        complete_output = complete_output
        complete_outputs.append(complete_output)
        complete_output = ''
    return complete_outputs

    
# Retrieval with Langchain 
    k=10
    model_kwargs = {'device':torch.device('cuda')}
    encode_kwargs = {'normalize_embeddings':True}
    
    hugging_face_embeddings = HuggingFaceEmbeddings(
        model_name = retrieval_config['pretrained_model_name_or_path'],
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )

    distance_strategy = DistanceStrategy.COSINE
    
    vector_db = FAISS.from_documents(
        documents = documents,
        embedding = hugging_face_embeddings,
        distance_strategy = distance_strategy
        )

    top_documents2 = vector_db.similarity_search_with_score(query, k=k)
    top_documents3 = vector_db.similarity_search(query, k=k)

    for top_document2 in top_documents2:
        print(f'{top_document2}\n')



    equal = []
    for i in range(k):
        if top_documents[i]['docs_content'] == top_documents2[i][0].page_content:
            equal.append((top_documents[i],top_documents2[i]))
    print(len(equal))
