import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

def setup_pinecone_index():
    # Create an instance of the Pinecone class
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )

    index_name = "hackrx-documents"
    dimension = 384
    
    try:
        # Check if index exists
        existing_indexes = pc.list_indexes()
        index_names = [index.name for index in existing_indexes.indexes]
        
        if index_name not in index_names:
            print(f"Creating new index: {index_name} with dimension {dimension}")
            # Create index using ServerlessSpec (recommended for new projects)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print(f"✅ Successfully created index: {index_name}")
        else:
            print(f"✅ Index '{index_name}' already exists")
            
        # Get the index object from the client instance
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"📊 Index stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error setting up Pinecone: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    setup_pinecone_index()