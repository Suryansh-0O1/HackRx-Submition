import os
import requests
import tempfile
import uvicorn
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

        # Load and split the PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load_and_split()
        os.unlink(temp_file_path)  # Clean up temp file

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Connect to existing Pinecone index and add documents
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Get existing index
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Create vector store from existing index and add new documents
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        # Add the new documents to the existing index
        vector_store.add_documents(chunks)

        # Create retriever with better parameters
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
        )

        # Initialize LLM with better parameters
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,  # Lower temperature for more consistent answers
            max_tokens=1000
        )

        # Create QA chain with better prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False,
            chain_type="stuff"  # Explicitly set chain type
        )
        
        return qa_chain

    except Exception as e:
        print(f"An error occurred in create_qa_chain: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process the document: {str(e)}")


# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "HackRX API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pinecone_configured": bool(PINECONE_API_KEY),
        "hackrx_key_configured": bool(HACKRX_API_KEY),
        "google_api_configured": bool(os.getenv("GOOGLE_API_KEY"))
    }

# Add both possible endpoint paths to be safe
@app.post("/hackrx/run", response_model=AnswerPayload, dependencies=[Depends(verify_api_key)])
@app.post("/api/v1/hackrx/run", response_model=AnswerPayload, dependencies=[Depends(verify_api_key)])
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

# Define the same function for the /api/v1/hackrx/run endpoint
async def run_hackathon_submission_v1(payload: QueryPayload):
    """Same function but for the v1 API endpoint"""
    return await run_hackathon_submission(payload)

# For running locally and on Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
else:
    # This helps Railway detect the app
    application = app
