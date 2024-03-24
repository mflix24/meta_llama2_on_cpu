from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from src.helper import *
from flask import Flask, render_template, jsonify, request


app = Flask(__name__)


# Step-1 : Load the datasetPDF File
loader=DirectoryLoader(
    'data/',
    glob="*.pdf",
    loader_cls=PyPDFLoader
    )
documents=loader.load()


# Step-2 : Splitting Text into Chunks through RecursiveCharacterTextSplitter()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
    )
text_chunks=text_splitter.split_documents(documents)


# Step-3 : Load the Embedding Model from HuggingFaceEmbeddings()
embeddings=HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2', 
    model_kwargs={'device':'cpu'}
    )


# Step-4 : Convert the Text Chunks into Embeddings and 
# storing the embeddings in a FAISS Vector Store
vector_store=FAISS.from_documents(text_chunks, embeddings)


# Step-5 : creating the LLM models through CTransformers() and passing the model name
llm=CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens':128, 'temperature':0.01}
    )


# Step-6 : creating prompt template through PromptTemplate() and passing few parametrs
# exporting template from src/helper.py
qa_prompt=PromptTemplate(
    template=template, 
    input_variables=['context', 'question']
    )


# Step-7 : Chaining theoug Langchain
# creating chain through RetrievalQA class and calling a function 
# by RetrievalQA class and passing few parameters.
# chaining the all components we just created.
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=False,
    chain_type_kwargs={'prompt': qa_prompt}
    )


# we are using flask here for UI design purposes.
# we are creating a function for index page as index.html
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

# creating a folder as chatbot and creating a function for  chatbot response
@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST':
        user_input = request.form['question']
        print(user_input)
        result=chain({'query':user_input})
        print(f"Answer:{result['result']}")

    return jsonify({"response": str(result['result']) })


# this is how flask works by giving hotid and its port number
if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 8080,debug=False)