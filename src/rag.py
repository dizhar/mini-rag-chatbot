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
        # Set the API key directly for older versions
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Try the new client format first, fallback to old format
        try:
            self.openai_client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.use_new_client = True
        except Exception:
            # Fallback to old openai library format
            self.use_new_client = False
            
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
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.use_new_client:
                # New OpenAI client format
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
            else:
                # Old OpenAI library format
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                batch_embeddings = [item['embedding'] for item in response['data']]
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Find the most relevant chunks for a query."""
        if self.embeddings is None:
            return []
        
        # Generate embedding for the query
        if self.use_new_client:
            query_response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            query_embedding = np.array([query_response.data[0].embedding])
        else:
            query_response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            query_embedding = np.array([query_response['data'][0]['embedding']])
        
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
        
        # Generate response
        if self.use_new_client:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
    
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