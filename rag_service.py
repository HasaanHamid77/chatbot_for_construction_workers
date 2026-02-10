"""
High-quality RAG service using LangChain for document processing.
Optimized for construction documents (SOPs, safety manuals, equipment manuals).
"""
import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# LangChain 
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



class RAGService:
    """Handles high-quality document retrieval using LangChain."""
    
    def __init__(self, documents_path="./documents"):
        """Initialize and load documents."""
        print("Initializing embedding model...")
        self.embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        self.documents = []
        self.index = None
        
        # LangChain text splitter optimized for construction documents
        # Prioritizes section breaks, numbered procedures, and design topics
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks to preserve complete procedures
            chunk_overlap=250,  # More overlap to maintain context
            length_function=len,
            separators=[
                # Section/clause breaks (highest priority)
                "\n\n## ",          # Markdown section headers
                "\n\nSection ",     # Common in manuals
                "\n\nClause ",      # Legal/spec documents
                "\n\nArticle ",     # Some safety manuals
                "\n\nChapter ",     # Chapter breaks
                
                # Numbered procedures (second priority)
                "\n\nProcedure ",   # Procedure headings
                "\n\nStep ",        # Step-by-step instructions
                "\n\n1. ",          # Numbered lists (procedures)
                "\n\n1) ",          # Alternative numbering
                
                # Design topics (third priority)
                "\n\nDesign ",      # Design sections
                "\n\nSpecification ", # Spec sections
                "\n\nRequirement ", # Requirements
                "\n\nGuideline ",   # Guidelines
                
                # Standard breaks (fallback)
                "\n\n\n",           # Multiple line breaks
                "\n\n",             # Paragraph breaks
                "\n",               # Line breaks
                ". ",               # Sentences
                "! ",
                "? ",
                "; ",
                ": ",
                " ",                # Words
                ""                  # Characters (last resort)
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
    
    def _extract_section_metadata(self, text):
        """Extract section/clause information from text for better organization."""
        metadata = {}
        
        # Try to extract section numbers (e.g., "Section 3.2.1", "Clause 5.4")
        section_match = re.search(r'(Section|Clause|Article|Chapter)\s+(\d+(?:\.\d+)*)', text[:200])
        if section_match:
            metadata['section_type'] = section_match.group(1)
            metadata['section_number'] = section_match.group(2)
        
        # Try to extract procedure types
        proc_match = re.search(r'(Procedure|Step|Instruction)\s+(\d+)', text[:200])
        if proc_match:
            metadata['procedure_type'] = proc_match.group(1)
            metadata['procedure_number'] = proc_match.group(2)
        
        # Try to extract design topics
        design_keywords = ['Design', 'Specification', 'Requirement', 'Guideline', 'Standard']
        for keyword in design_keywords:
            if keyword.lower() in text[:300].lower():
                metadata['topic_type'] = keyword
                break
        
        return metadata
    
    def process_documents(self, documents_path):
        """Process all documents using LangChain loaders."""
        if not os.path.exists(documents_path):
            print(f"No documents folder found at {documents_path}")
            self.index = faiss.IndexFlatL2(384)
            return
        
        all_docs = []
        
        for filename in os.listdir(documents_path):
            filepath = os.path.join(documents_path, filename)
            
            # Skip hidden files and non-document files
            if filename.startswith('.') or os.path.isdir(filepath):
                continue
            
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
                    
                    # Determine document type from filename
                    filename_lower = filename.lower()
                    if 'sop' in filename_lower or 'procedure' in filename_lower:
                        doc.metadata['doc_type'] = 'SOP'
                    elif 'safety' in filename_lower or 'hazard' in filename_lower:
                        doc.metadata['doc_type'] = 'Safety Manual'
                    elif 'equipment' in filename_lower or 'manual' in filename_lower:
                        doc.metadata['doc_type'] = 'Equipment Manual'
                    elif 'spec' in filename_lower or 'requirement' in filename_lower:
                        doc.metadata['doc_type'] = 'Specification'
                    else:
                        doc.metadata['doc_type'] = 'General'
                
                all_docs.extend(docs)
                print(f"Loaded {filename}: {len(docs)} pages/sections ({doc.metadata.get('doc_type', 'General')})")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        if not all_docs:
            print("No documents to process")
            self.index = faiss.IndexFlatL2(384)
            return
        
        # Split documents into chunks using intelligent chunking
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(all_docs)
        print(f"Created {len(chunks)} chunks from {len(all_docs)} document pages")
        
        # Extract text and metadata
        texts = []
        metadata_list = []
        
        for chunk in chunks:
            # Extract additional metadata from chunk content
            section_meta = self._extract_section_metadata(chunk.page_content)
            
            # Combine all metadata
            combined_metadata = {
                'text': chunk.page_content,
                'document': chunk.metadata.get('source_file', 'unknown'),
                'page': chunk.metadata.get('page', chunk.metadata.get('page_number')),
                'source': chunk.metadata.get('source', ''),
                'doc_type': chunk.metadata.get('doc_type', 'General'),
                **section_meta  # Add extracted section/clause metadata
            }
            
            texts.append(chunk.page_content)
            metadata_list.append(combined_metadata)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embeddings.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(384)
        self.index.add(np.array(embeddings).astype('float32'))
        
        self.documents = metadata_list
        print(f"✓ RAG ready with {len(self.documents)} chunks")
        
        # Print document type summary
        doc_types = {}
        for doc in self.documents:
            doc_type = doc.get('doc_type', 'General')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"\n📊 Document breakdown:")
        for doc_type, count in sorted(doc_types.items()):
            print(f"  • {doc_type}: {count} chunks")
    
    def search(self, query, top_k=4):
        """Search for relevant documents with similarity scoring and metadata."""
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
            
            # Build comprehensive context with metadata
            source_info = f"{doc['document']}"
            
            # Add page if available
            if doc.get('page'):
                source_info += f", Page {doc['page']}"
            
            # Add section/clause if available
            if doc.get('section_type') and doc.get('section_number'):
                source_info += f", {doc['section_type']} {doc['section_number']}"
            
            # Add document type
            if doc.get('doc_type') and doc['doc_type'] != 'General':
                source_info += f" ({doc['doc_type']})"
            
            # Format context with rich metadata
            context_parts.append(
                f"[Source {i+1}: {source_info}]\n{doc['text']}"
            )
            
            citations.append({
                'document': doc['document'],
                'page': doc.get('page'),
                'section': f"{doc.get('section_type', '')} {doc.get('section_number', '')}".strip(),
                'doc_type': doc.get('doc_type'),
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
