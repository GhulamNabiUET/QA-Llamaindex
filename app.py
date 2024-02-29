import os
import openai
import tiktoken
import streamlit as st


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptHelper
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
# from langchain.chat_models import ChatOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# OPENAI_API_KEY = " "
os.environ["OPENAI_API_KEY"]=os.getenv('OPENAI_API_KEY')

# apenai_api_key = st.secrets("OPENAI_API_KEY")

llm = OpenAI(model='gpt-4-turbo-preview', temperature=0, max_tokens=256)
embed_model = OpenAIEmbedding()

# Streamlit UI
st.title("Trader-GPT")

# Specify the directory containing the CSV files
input_directory = 'Dataset_'
# Create an instance of SimpleDirectoryReader pointing to the input directory
documents_reader = SimpleDirectoryReader(input_directory)
# Load the documents (CSV files in this case)
documents = documents_reader.load_data()
# print(documents)

node_parser = SimpleNodeParser.from_defaults(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
  paragraph_separator="\n\n\n",
  secondary_chunking_regex="[^,.;。]+[,.;。]?",
  tokenizer=tiktoken.encoding_for_model("gpt-4-turbo-preview").encode
)

prompt_helper = PromptHelper(
  context_window=4096,
  num_output=256,
  chunk_overlap_ratio=0.1,
  chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(
  llm=llm,
  embed_model=embed_model,
  node_parser=node_parser,
  prompt_helper=prompt_helper
)

# Create an index from the documents
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Define a custom prompt for trading signal transformation
template = (
    "Transform the trading signal input into a structured JSON format as shown in the examples below. \n"
    "---------------------\n"
    "Example 1:\n"
    "Input: HIGH RISK SPX here 01/02 4730P Avg. 3.30\n"
    "Output: {{ \"action\": \"BTO\", \"ticker\": \"SPX\", \"strike\": 4730, \"right\": \"P\", \"entry\": 3.30, \"size\": \"complete\" }}\n"
    "\n"
    "Example 2:\n"
    "Input: Riskier SPX here 01/04 4720C Avg. 3.00 Stay light\n"
    "Output: {{ \"action\": \"BTO\", \"ticker\": \"SPX\", \"strike\": 4720, \"right\": \"C\", \"entry\": 3.00, \"size\": \"complete\" }}\n"
    "---------------------\n"
    "Please transform the following input into the structured JSON format as per the examples given:\n"
    "{context_str}\n"
    "Given this information, please answer the question and each answer should start with code word AI Demos: {query_str}\n"
)
qa_template = PromptTemplate(template)

# Create a query engine from the index
query_engine = index.as_query_engine(text_qa_template=qa_template)

# Adjust your query to fit the new format. Example:
# input_signal = "HIGH RISK SPX here 01/02 4730P Avg. 3.30"
# response = query_engine.query(input_signal)
# print(response)

# User input
user_query = st.text_input("Enter your question:", "")

# Query the engine
# response = query_engine.query("HIGH RISK SPX here 01/02 4730P Avg. 3.30")
# print(response)

if st.button("Get Answer"):
    # Query the engine
    response = query_engine.query(user_query)
    response_text = response.response if hasattr(response, 'response') else "No valid response."
    st.write(response_text)
