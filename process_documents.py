from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# python-dotenv reads key-value pairs from a .env file and can set them as environment variables. 
from dotenv import load_dotenv
#Miscellaneous operating system interfaces
import os
#Miscellaneous operating system interfaces
import shutil



#Load environment variables
load_dotenv()


#Loads PDFs from the specified director, then breaks them up into chunks
def load_and_process_pdfs(data_dir: str):
   
   
    loader = DirectoryLoader(
        # takes inputted directory string.
        data_dir,
        #all pdfs in the folder.
        glob="**/*.pdf",
        #use document loader for PDF files.
        loader_cls=PyPDFLoader
    )
    
    #assigns the loader.load() return to the documents variable.
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        #Chunk size is the maximum number of characters that a chunk can contain.
        chunk_size=1000,
        #How many characters should overlap between chunks
        chunk_overlap=200,
        #pyhton len() function
        length_function=len,
    )
    
    #Use text_splitter object to split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    return chunks




#Creating a chroma vector store - https://docs.trychroma.com/docs/overview/introduction
def create_vector_store(chunks, persist_directory: str):

    # Clear existing vector store if it exists
    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # Initialize HuggingFace embeddings
    # https://huggingface.co/blog/getting-started-with-embeddings - turns data into vectors
    embeddings = HuggingFaceEmbeddings(
        #this model maps text/paragraphs to vector space - https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create and persist Chroma vector store
    print("Creating new vector store...")
    vectordb = Chroma.from_documents(
        # the processed chunks from our data (pdf)
        documents=chunks,
        # the vector handler
        embedding=embeddings,
        # the directory location
        persist_directory=persist_directory
    )
    
    #returns the vector database
    return vectordb





#main method
def main():
    

    # Define directories
    # __file__ is a built-in constant containing the pathname of the file from which the running Python module was loaded
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    # Process PDFs
    print("Loading and processing PDFs...")
    chunks = load_and_process_pdfs(data_dir)
    print(f"Created {len(chunks)} chunks from PDFs")
    
    # Create vector store
    print("Creating vector store...")
    vectordb = create_vector_store(chunks, db_dir)
    print(f"Vector store created and persisted at {db_dir}")

# __name__ is a variable that exists in every Python module, and is set to the name of the module
if __name__ == "__main__":
    main()