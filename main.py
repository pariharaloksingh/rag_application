from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Step 1: Load and prepare data
loader = TextLoader("doc.txt", encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=60)
texts = text_splitter.split_documents(documents)

# Step 2: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3: Build vector store
vectorstore = FAISS.from_documents(texts, embeddings)

# Step 4: Load language model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Step 5: Create RAG pipeline
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Step 6: Query the RAG application
query = "What is current?"
response = qa_chain.run(query)
print(response)