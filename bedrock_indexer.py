# from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import BedrockEmbeddings
# from langchain_aws import BedrockEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# def index_directory(directory_path, glob_pattern="**/[!.]*", chunk_size=500):
#     # Initialize Bedrock embeddings
#     embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")
#     # Load documents from directory
#     loader = DirectoryLoader(
#         directory_path, 
#         glob=glob_pattern, 
#         show_progress=True, 
#         use_multithreading=True
#     )
#     documents = loader.load()

#     # Use RecursiveCharacterTextSplitter for better chunk handling
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=50,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     docs = text_splitter.split_documents(documents)

#     # Create and save FAISS vectorstore
#     # return FAISS.from_documents(docs, embeddings).save_local("faiss_index")
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     vectorstore.save_local("faiss_index")
#     # print(f"Total documents indexed: {vectorstore.index.ntotal}")
#     return vectorstore
# # Example usage
# directory_path = "documents/"
# vectorstore = index_directory(directory_path)
# print(vectorstore.index.ntotal)





### S3 as a Document store with improved error handling
import boto3
import os
import logging
from typing import List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import signal
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def timeout(duration):
    """Context manager for timeout functionality"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {duration} seconds")
    
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)  # Set alarm
    
    try:
        yield
    finally:
        signal.alarm(0)  # Disable alarm

def load_single_docx(local_path: str, max_timeout: int = 30) -> List[Document]:
    """Load a single DOCX file with timeout and error handling"""
    documents = []
    try:
        logger.info(f"Processing: {os.path.basename(local_path)}")
        
        with timeout(max_timeout):
            loader = UnstructuredWordDocumentLoader(local_path)
            documents = loader.load()
            
        logger.info(f"Successfully loaded {len(documents)} documents from {os.path.basename(local_path)}")
        
    except TimeoutError:
        logger.error(f"Timeout while processing {os.path.basename(local_path)} - skipping")
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(local_path)}: {str(e)} - skipping")
    
    return documents

def load_docx_from_s3(bucket_name: str, prefix: str = "", max_files: int = None) -> List[Document]:
    """Load DOCX files from S3 with improved error handling"""
    s3 = boto3.client("s3")
    
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    except Exception as e:
        logger.error(f"Error listing S3 objects: {str(e)}")
        return []
    
    documents = []
    docx_files = []
    
    # Collect all DOCX files
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".docx") and not key.startswith("~$"):  # Skip temp files
            docx_files.append((key, obj.get("Size", 0)))
    
    # Sort by size (smaller files first) to process easier files first
    docx_files.sort(key=lambda x: x[1])
    
    # Limit number of files if specified
    if max_files:
        docx_files = docx_files[:max_files]
    
    logger.info(f"Found {len(docx_files)} DOCX files to process")
    
    for i, (key, size) in enumerate(docx_files, 1):
        logger.info(f"Processing file {i}/{len(docx_files)}: {key} ({size} bytes)")
        
        local_path = f"/tmp/{os.path.basename(key)}"
        
        try:
            # Download file
            s3.download_file(bucket_name, key, local_path)
            
            # Process file with timeout
            file_documents = load_single_docx(local_path, max_timeout=60)
            documents.extend(file_documents)
            
        except Exception as e:
            logger.error(f"Error downloading {key}: {str(e)} - skipping")
        
        finally:
            # Clean up local file
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {local_path}: {str(e)}")
    
    logger.info(f"Successfully loaded {len(documents)} total documents")
    return documents

def index_s3_directory(bucket_name: str, prefix: str = "", chunk_size: int = 500, max_files: int = None):
    """Index S3 directory with improved error handling"""
    logger.info(f"Starting indexing process for bucket: {bucket_name}, prefix: {prefix}")
    
    try:
        # Create Bedrock embeddings with explicit region specification
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name="us-east-1"
        )
        logger.info("Bedrock embeddings initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing Bedrock embeddings: {str(e)}")
        return None
    
    # Load documents from S3
    documents = load_docx_from_s3(bucket_name, prefix, max_files)
    
    if not documents:
        logger.warning("No documents loaded - cannot create vector store")
        return None
    
    logger.info(f"Splitting {len(documents)} documents into chunks")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    try:
        docs = text_splitter.split_documents(documents)
        logger.info(f"Created {len(docs)} document chunks")
        
        # Create vector store
        logger.info("Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Save vector store
        vectorstore.save_local("faiss_index")
        logger.info("Vector store saved successfully")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

# -------- Entry Point --------
if __name__ == "__main__":
    bucket_name = "redington-presales-bot-2025"
    prefix = "proposals/"  # For all files, or specify folder like "docs/"
    
    # Start with a limited number of files for testing
    max_files = 50  # Remove or set to None to process all files
    
    logger.info("Starting S3 document indexing process")
    
    vectorstore = index_s3_directory(bucket_name, prefix, max_files=max_files)
    
    if vectorstore:
        print(f"Successfully created vector store with {vectorstore.index.ntotal} vectors")
        logger.info("Indexing process completed successfully")
    else:
        print("Failed to create vector store")
        logger.error("Indexing process failed")
