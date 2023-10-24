# content_creator
Fine tuning Llama2 on specific website content

The `.ipynb` file will open in Collab if you would like to try it out. I have just pasted the Collab source code below for readability (i.e. the following runs in my Collab):

```bash
!pip install kaleido
!pip install langchain
!pip install huggingface_hub
!pip install sentence_transformers
!pip install faiss-cpu
!pip install unstructured
!pip install chromadb
!pip install Cython
!pip install tiktoken
!pip install unstructured[local-inference]
!pip install -qU transformers 
!pip install -qU accelerate 
!pip install -qU einops 
!pip install -qU xformers 
!pip install -qU bitsandbytes 
!pip install -qU faiss-gpu
```

Restart the Collab runtime. The run the following code:

```python
# Generate and then add your HuggingFace API Token in here
YourHuggingFaceAPIToken = 'abcd'
# Imports
import os
import torch
import requests
import transformers
from torch import cuda, bfloat16
import xml.dom.minidom as minidom
from langchain.document_loaders import TextLoader 
from langchain.text_splitter import CharacterTextSplitter 
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import WebBaseLoader
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Fetch sitemap's text of each site that is to be indexed
fermyon_website_sitemap = requests.get('https://www.fermyon.com/sitemap.xml', allow_redirects=True).text
fermyon_documentation_sitemap = requests.get('https://developer.fermyon.com/sitemap.xml', allow_redirects=True).text
component_model_documentation_sitemap = requests.get('https://component-model.bytecodealliance.org/sitemap.xml', allow_redirects=True).text

# Parse each sitemap's text to obtain list of pages
parsed_fermyon_website_sitemap_document = minidom.parseString(fermyon_website_sitemap)
parsed_fermyon_documentation_sitemap_document = minidom.parseString(fermyon_documentation_sitemap)
parsed_component_model_documentation_sitemap_document = minidom.parseString(component_model_documentation_sitemap)

# Cherry pick just the loc elements from the XML
fermyon_website_sitemap_loc_elements = parsed_fermyon_website_sitemap_document.getElementsByTagName('loc')
fermyon_documentation_sitemap_loc_elements = parsed_fermyon_documentation_sitemap_document.getElementsByTagName('loc')
component_model_documentation_sitemap_loc_elements = parsed_component_model_documentation_sitemap_document.getElementsByTagName('loc')

# Declare blank lists of pages for each site
fermyon_website_page_urls = []
fermyon_documentation_page_urls = []
component_model_documentation_page_urls = []

# Iterate over loc elements (of each sitemap) and add to that site's list of pages
for fermyon_website_sitemap_loc_element in fermyon_website_sitemap_loc_elements:
    fermyon_website_page_urls.append(fermyon_website_sitemap_loc_element.toxml().removesuffix("</loc>").removeprefix("<loc>"))
for fermyon_documentation_sitemap_loc_element in fermyon_documentation_sitemap_loc_elements:
    fermyon_documentation_page_urls.append(fermyon_documentation_sitemap_loc_element.toxml().removesuffix("</loc>").removeprefix("<loc>"))
for component_model_documentation_sitemap_loc_element in component_model_documentation_sitemap_loc_elements:
    component_model_documentation_page_urls.append(component_model_documentation_sitemap_loc_element.toxml().removesuffix("</loc>").removeprefix("<loc>"))

URLs = fermyon_website_page_urls + fermyon_documentation_page_urls + component_model_documentation_page_urls

text_to_remove = "rss"
filtered_list = [item for item in URLs if text_to_remove not in item]

print("Number of page to process is {}\n First page to process is {} and the last page to process is {}".format(len(filtered_list), filtered_list[0], filtered_list[len(filtered_list) - 1]))

os.environ["HUGGINGFACEHUB_API_TOKEN"] = YourHuggingFaceAPIToken
hf_auth = YourHuggingFaceAPIToken
loader2 = [UnstructuredURLLoader(urls=filtered_list)]
model_id = 'meta-llama/Llama-2-7b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
!pip install accelerate
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

print(f"Model loaded on {device}")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)
stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])
generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
llm = HuggingFacePipeline(pipeline=generate_text)
web_links = filtered_list
loader = WebBaseLoader(web_links)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)
from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)
```

After the above has run in the Collab, you can test by running the following code (insert your own questions that relate the the websites that were indexed above):

```python
chat_history = []

query1 = "Can you please write me a few sentences about what Fermyon does and why a developer would want to use Fermyon Spin and Fermyon Cloud?"
result1 = chain({"question": query1, "chat_history": chat_history})

print("Tell me about Fermyon?\n")
print(result1['answer'])
```
