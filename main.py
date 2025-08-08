import os
import requests
import tempfile
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Pinecone components
from langchain_pinecone import PineconeVectorStore


# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# --- Pydantic Models for Request and Response ---
class QueryPayload(BaseModel):
    """Defines the expected JSON structure for incoming requests."""
    documents: str
    questions: list[str]

class AnswerPayload(BaseModel):
    """Defines the JSON structure for the response."""
    answers: list[str]

# --- Authentication ---
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY")
security = HTTPBearer()

async def verify_api_key(token: HTTPAuthorizationCredentials = Depends(security)):
    """This dependency verifies the Bearer token in the Authorization header."""
    if token.credentials != HACKRX_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# --- Pinecone Initialization ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hackrx-documents"  # Updated to match the index name in create_pinecone_index.py

# --- RAG Logic ---
def create_qa_chain(pdf_url: str):
    """
    This function takes a URL to a PDF, downloads it, adds its content to the
    Pinecone index, and returns a RetrievalQA chain.
    """
    try:
        # Download the PDF from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split()
        os.unlink(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load the existing index and add the new documents
        # This will add the new PDF's content to your index without deleting old content.
        vector_store = PineconeVectorStore.from_documents(
            chunks, embeddings, index_name=PINECONE_INDEX_NAME
        )

        retriever = vector_store.as_retriever()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )
        return qa_chain

    except Exception as e:
        print(f"An error occurred in create_qa_chain: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the document.")


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=AnswerPayload, dependencies=[Depends(verify_api_key)])
async def run_hackathon_submission(payload: QueryPayload):
    """
    This is the main endpoint for the hackathon. It receives a document URL
    and a list of questions, processes them, and returns the answers.
    """
    pdf_url = payload.documents
    questions = payload.questions
    answers = []

    try:
        qa_chain = create_qa_chain(pdf_url)
        for question in questions:
            if question:
                result = qa_chain.invoke(question)
                answers.append(result.get('result', 'Could not find an answer.'))
            else:
                answers.append("Invalid question provided.")
        return {"answers": answers}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")