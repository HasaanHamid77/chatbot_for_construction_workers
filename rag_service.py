"""
High-quality RAG service using LangChain for document processing.
"""
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# LangChain 
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


class RAGService:
    """Handles high-quality document retrieval using LangChain."""
    
    def __init__(self, documents_path="./documents"):
        """Initialize and load documents."""
        print("Initializing embedding model...")
        self.embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.documents = []
        self.index = None
        
        # LangChain text splitter for intelligent chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Larger chunks for better context
            chunk_overlap=200,  # Significant overlap to preserve context
            length_function=len,
            separators=[
                "\n\n",      # Paragraph breaks (highest priority)
                "\n",        # Line breaks
                ". ",        # Sentences
                "! ",
                "? ",
                "; ",
                ": ",
                " ",         # Words
                ""           # Characters (last resort)
            ],
            is_separator_regex=False
        )
        
        # Load or create index
        if os.path.exists("faiss.index") and os.path.exists("documents.pkl"):
            print("Loading existing index...")
            self.load_index()
        else:
            print("Processing documents...")
            self.process_documents(documents_path)
            self.save_index()
    
    def process_documents(self, documents_path):
        """Process all documents using LangChain loaders."""
        if not os.path.exists(documents_path):
            print(f"No documents folder found at {documents_path}")
            self.index = faiss.IndexFlatL2(384)
            return
        
        all_docs = []
        
        for filename in os.listdir(documents_path):
            filepath = os.path.join(documents_path, filename)
            
            try:
                # Use appropriate LangChain loader
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                elif filename.endswith('.docx'):
                    loader = Docx2txtLoader(filepath)
                elif filename.endswith('.txt'):
                    loader = TextLoader(filepath, encoding='utf-8')
                else:
                    print(f"Skipping unsupported file: {filename}")
                    continue
                
                # Load documents
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata['source_file'] = filename
                
                all_docs.extend(docs)
                print(f"Loaded {filename}: {len(docs)} pages/sections")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        if not all_docs:
            print("No documents to process")
            self.index = faiss.IndexFlatL2(384)
            return
        
        # Split documents into chunks using LangChain
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(all_docs)
        print(f"Created {len(chunks)} chunks from {len(all_docs)} documents")
        
        # Extract text and metadata
        texts = []
        metadata_list = []
        
        for chunk in chunks:
            texts.append(chunk.page_content)
            metadata_list.append({
                'text': chunk.page_content,
                'document': chunk.metadata.get('source_file', 'unknown'),
                'page': chunk.metadata.get('page', chunk.metadata.get('page_number')),
                'source': chunk.metadata.get('source', '')
            })
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embeddings.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(384)
        self.index.add(np.array(embeddings).astype('float32'))
        
        self.documents = metadata_list
        print(f"✓ RAG ready with {len(self.documents)} chunks")
    
    def search(self, query, top_k=4):
        """Search for relevant documents with similarity scoring."""
        if len(self.documents) == 0:
            return "", []
        
        # Generate query embedding
        query_embedding = self.embeddings.encode([query])[0]
        
        # Search with scores
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            min(top_k, len(self.documents))
        )
        
        # Build context and citations
        context_parts = []
        citations = []
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx >= len(self.documents):
                continue
            
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + distance)
            
            # Filter out low-relevance results (similarity < 0.3)
            if similarity < 0.3:
                continue
            
            doc = self.documents[idx]
            
            # Format context with document reference
            page_info = f", Page {doc['page']}" if doc.get('page') else ""
            context_parts.append(
                f"[Source {i+1}: {doc['document']}{page_info}]\n{doc['text']}"
            )
            
            citations.append({
                'document': doc['document'],
                'page': doc.get('page'),
                'text': doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'],
                'relevance_score': float(similarity)
            })
        
        context = "\n\n".join(context_parts)
        return context, citations
    
    def save_index(self):
        """Save FAISS index and documents."""
        if self.index is not None and len(self.documents) > 0:
            faiss.write_index(self.index, "faiss.index")
            with open("documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            print("✓ Index saved")
    
    def load_index(self):
        """Load FAISS index and documents."""
        self.index = faiss.read_index("faiss.index")
        with open("documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        print(f"✓ Index loaded with {len(self.documents)} chunks")