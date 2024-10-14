from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX_NAME="langchain-doc-index"

def run_llm(query:str):
    embeddings = OllamaEmbeddings(model="llama3.2")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOllama(verbose = True, temperature = 0, model="llama3.2")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    res = run_llm(query="What is a LangChain?")
    print(res["answer"])
