"""Core RAG functionality for the chatbot."""

import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv

load_dotenv()


class RAGChatbot:
    def __init__(self):
        # Initialize OpenAI client (v1.0+ format)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.chunks = []
        self.embeddings = None
        self.metadata = []
    
    def load_documents(self, chunks_with_metadata: List[Dict]):
        """Load processed documents into the RAG system."""
        self.chunks = [chunk['text'] for chunk in chunks_with_metadata]
        self.metadata = chunks_with_metadata
        
        print(f"Loaded {len(self.chunks)} chunks from documents")
        print("Generating embeddings...")
        
        # Generate embeddings for all chunks
        self.embeddings = self._generate_embeddings(self.chunks)
        print("✅ Embeddings generated successfully!")
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 50  # Reduced batch size for stability
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise e
        
        return np.array(embeddings)
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Find the most relevant chunks for a query."""
        if self.embeddings is None:
            return []
        
        try:
            # Generate embedding for the query
            query_response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            query_embedding = np.array([query_response.data[0].embedding])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append((
                    self.chunks[idx],
                    self.metadata[idx],
                    similarities[idx]
                ))
            
            return results
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def generate_answer(self, query: str, relevant_chunks: List[Tuple[str, Dict, float]]) -> str:
        """Generate an answer using OpenAI with relevant context."""
        # Prepare context from relevant chunks
        context_parts = []
        for chunk, metadata, score in relevant_chunks:
            context_parts.append(f"From {metadata['source']}, Page {metadata['page']}:\n{chunk}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant answering questions about company policies. 
Use the provided context to answer the user's question. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def ask(self, question: str) -> Dict:
        """Main method to ask a question and get an answer with sources."""
        # Find relevant chunks
        relevant_chunks = self.search_relevant_chunks(question, top_k=3)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, relevant_chunks)
        
        # Prepare sources for display
        sources = []
        for chunk, metadata, score in relevant_chunks:
            sources.append({
                "source": metadata['source'],
                "page": metadata['page'],
                "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "relevance_score": float(score)
            })
        
        return {
            "answer": answer,
            "sources": sources
        }