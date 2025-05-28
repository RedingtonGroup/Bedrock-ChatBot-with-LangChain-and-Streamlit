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





### S3 as a Document store
import boto3
import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_docx_from_s3(bucket_name, prefix=""):
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    documents = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".docx"):
            local_path = f"/tmp/{os.path.basename(key)}"
            s3.download_file(bucket_name, key, local_path)
            loader = UnstructuredWordDocumentLoader(local_path)
            documents.extend(loader.load())
            os.remove(local_path)  # Clean up
    return documents

def index_s3_directory(bucket_name, prefix="", chunk_size=500):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1")
    
    documents = load_docx_from_s3(bucket_name, prefix)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# -------- Entry Point --------
bucket_name = "redington-presales-bot-2025"
prefix = ""  # For all files, or specify folder like "docs/"
vectorstore = index_s3_directory(bucket_name, prefix)
print(vectorstore.index.ntotal)
