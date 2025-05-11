from langchain_community.tools import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langgraph.graph import Graph, END
import os
import api

query = "What are superconductors?"
HUGGINGFACE_API_KEY = "your_api_key"
os.environ["TAVILY_API_KEY"] = "Your_api_key"

# Initialize Tavily Search Tool for performing web searches
tavily_search = TavilySearchResults()

# Create a wrapper for the SentenceTransformer model so that it fits the interface expected by LangChain.
# It is a important step as directly using model gives error
class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode(text)

    def __call__(self, text):
        # Allow the instance to be called as a function
        return self.embed_query(text)

# Initialize your SentenceTransformer model as before and Convert model to FP16
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cuda")
embedding_model.half()  

# Wrap the model
embedding_wrapper = SentenceTransformerWrapper(embedding_model)
# Load Hugging Face text generation model
hf_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

def research_agent(state):
    query = state["query"]

    # Load detailed web data
    loader = TavilySearchResults()
    documents = loader.run(query)

    # Extract page content from dictionaries
    texts = [doc["content"] for doc in documents if "content" in doc]

    # Split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts)

    # Initialize FAISS vector store correctly with the wrapped model
    vector_db = FAISS.from_texts([doc.page_content for doc in docs], embedding_wrapper)

    # Store the created vector database in the state so that the next agent can access it.
    state["vector_db"] = vector_db
    return state


def answer_drafting_agent(state):
    query = state["query"]
    vector_db = state["vector_db"]
    
    # The retriever is used to search and retrieve documents most relevant to the query.
    retriever = vector_db.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(query)

    # Concatenate retrieved texts
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    input_prompt = f""" Examine the following context and extract its essential details.  
    Using this information, craft a clear and comprehensive answer to the query.  
    Ensure every sentence is complete and that no sentence is abruptly truncated.  
    Only answer the user’s queried question without adding unrelated details.  
    Organize your response logically and concisely, removing any extraneous information, 
    and answer provided by you should be of summary type :\n\n{context}\n\nQuery: {query}\nAnswer:"""
    
    response = hf_generator(input_prompt, max_length=1000, do_sample=True)[0]["generated_text"]
    
    state["response"] = response
    return state

# Define Workflow
workflow = Graph()

# Add Nodes for the Agents
workflow.add_node("ResearchAgent", research_agent)
workflow.add_node("AnswerDraftingAgent", answer_drafting_agent)

# Connect Nodes to define the Workflow
workflow.add_edge("ResearchAgent", "AnswerDraftingAgent")
workflow.add_edge("AnswerDraftingAgent", END)

# Setting Entry Point
workflow.set_entry_point("ResearchAgent")

# Compile Workflow
chain = workflow.compile()

# Run the Workflow
result = chain.invoke({"query": query})

print("\nGenerated Response:\n", result["response"])