from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.schema import Document

# Initialize Flask app
app = Flask(__name__)
from flask import Flask

app = Flask(__name__)

# ✅ Add a route for the homepage
@app.route("/")
def home():
    return "Chatbot API is running!"

# ✅ Add a route to handle favicon requests
@app.route("/favicon.ico")
def favicon():
    return "", 204  # No Content response

if __name__ == "__main__":
    app.run(debug=True)


# Load the embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load ChromaDB
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Retrieve relevant documents
    results = db.similarity_search(user_query, k=3)

    # Prepare response
    response_text = "\n".join([doc.page_content for doc in results])

    return jsonify({"query": user_query, "response": response_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # ❌ Disable debug mode'''

'''import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain

# ✅ Set OpenAI API Key (Replace with your key)
os.environ["OPENAI_API_KEY"] = "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"

# ✅ Load OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# ✅ Load vector database correctly
embedding_model = OpenAIEmbeddings()
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# ✅ Ensure `llm` is not None
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def chat_with_bot(question):
    response = qa.invoke({"question": question, "chat_history": []})
    return response["answer"]

if __name__ == "__main__":
    print("Chatbot is running! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = chat_with_bot(user_input)
        print("Chatbot:", response)'''
'''import os
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from transformers import pipeline  # Hugging Face's transformer pipeline

# Define paths and model names
model_name = "gpt2"  # Or another Hugging Face model if needed
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example for embeddings, change as needed
persist_directory = "db"  # Folder for Chroma database

# Initialize Hugging Face transformer pipeline for text generation
generator = pipeline("text-generation", model=model_name)

# Load Hugging Face model for embeddings
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Initialize the memory buffer to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load documents into the vector store (you can change this as needed)
loader = TextLoader("docs")  # Change to your folder or data source
documents = loader.load()

# Create the vector store with embeddings
vector_store = Chroma.from_documents(documents, embedding_function, persist_directory=persist_directory)

# Create a simple prompt template for interaction
prompt_template = """
You are a helpful assistant. You have access to the following context from the documents:
{context}

Your task is to respond to the user's query based on the context.

User Query: {query}
Assistant:
"""
prompt = PromptTemplate(input_variables=["context", "query"], template=prompt_template)

# Set up the LLMChain
llm_chain = LLMChain(prompt=prompt, llm=generator)

# Function to retrieve context and answer the query
def retrieve_context_and_answer(query):
    # Retrieve relevant documents
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query)

    # Prepare the context for the prompt
    context = "\n".join([doc.page_content for doc in docs])

    # Run the LLM chain to generate the answer
    response = llm_chain.run({"context": context, "query": query})
    return response

# Function to interact with the chatbot
def chatbot_query(query):
    response = retrieve_context_and_answer(query)
    return response

if __name__ == "__main__":
    print("Chatbot is ready! Start chatting with the bot:")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting the chatbot...")
            break
        response = chatbot_query(query)
        print(f"Bot: {response}")'''
