import requests
import os
import time
import argparse

import whisper
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import feedparser

class PodcastProcessor:
    def __init__(self, model_name="llama3", output_dir="podcast_data"):
        """
        Initialize the podcast processor using local Ollama models.
        
        Args:
            model_name (str): Name of the Ollama model to use
            output_dir (str): Directory to store downloaded files
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        self.transcript_dir = os.path.join(output_dir, "transcripts")
        self.db_dir = os.path.join(output_dir, "vectordb")
        
        # Create necessary directories
        for directory in [self.audio_dir, self.transcript_dir, self.db_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize Whisper model for transcription
        self.transcription_model = whisper.load_model("base")
        
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=model_name)
        
        # Make sure Ollama is running
        self._check_ollama_running()
        
    def _check_ollama_running(self):
        """Verify that Ollama is running and the model is available"""
        try:
            # Check if Ollama is running by making a request to the local API
            response = requests.get("http://localhost:11434/api/tags")
            available_models = [model["name"] for model in response.json()["models"]]
            
            if self.model_name not in available_models:
                print(f"Warning: Model '{self.model_name}' not found in Ollama. Available models: {available_models}")
                print(f"You may need to run: ollama pull {self.model_name}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Ollama service is not running. Start Ollama before running this script.")
    
    def extract_rss_from_apple_podcast(self, apple_podcast_url):
        """
        Extract the RSS feed URL from an Apple Podcasts URL.
        
        Args:
            apple_podcast_url (str): URL of the Apple Podcasts episode
            
        Returns:
            str: RSS feed URL
        """
        # This is a simplified implementation
        # In a real-world scenario, you'd need to scrape the Apple Podcasts page to get the RSS URL
        # For this example, we'll use the Empire podcast RSS feed directly
        return "https://feeds.megaphone.fm/empirepodcast"
    
    def download_podcast_audio(self, podcast_url, episode_id=None):
        """
        Download podcast audio from the given URL.
        
        Args:
            podcast_url (str): URL of the podcast episode
            episode_id (str, optional): ID to assign to the episode
            
        Returns:
            str: Path to the downloaded audio file
        """
        print(f"Downloading podcast from {podcast_url}")
        
        # Extract the RSS feed from the Apple Podcasts URL
        rss_url = self.extract_rss_from_apple_podcast(podcast_url)
        print(f"Found RSS feed: {rss_url}")
        
        # Parse the RSS feed
        feed = feedparser.parse(rss_url)
        
        # Find the episode in the feed (in a real implementation, you'd match by the episode ID in the URL)
        # For this example, we'll use the first episode
        episode = feed.entries[0]
        
        # Generate episode ID if not provided
        if episode_id is None:
            episode_id = f"episode_{int(time.time())}"
        
        # Find the audio file URL
        audio_url = None
        for link in episode.links:
            if link.get('type', '').startswith('audio/'):
                audio_url = link.href
                break
        
        if not audio_url:
            for enclosure in episode.get('enclosures', []):
                if enclosure.get('type', '').startswith('audio/'):
                    audio_url = enclosure.href
                    break
        
        if not audio_url:
            print("Could not find audio URL in the feed")
            # Create a placeholder file
            audio_file_path = os.path.join(self.audio_dir, f"{episode_id}.mp3")
            with open(audio_file_path, 'w') as f:
                f.write("Placeholder audio file")
            return audio_file_path
        
        # Download the audio file
        audio_file_path = os.path.join(self.audio_dir, f"{episode_id}.mp3")
        print(f"Downloading audio from {audio_url} to {audio_file_path}")
        
        # Download the file here
        response = requests.get(audio_url)
        with open(audio_file_path, 'wb') as f:
            f.write(response.content)
        
        
        return audio_file_path
    
    def generate_transcript(self, audio_file_path):
        """
        Generate transcript/closed captions from the audio file.
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            str: Path to the transcript file
        """
        print(f"Generating transcript for {audio_file_path}")
        
        # Extract episode ID from the filename
        episode_id = os.path.basename(audio_file_path).split('.')[0]
        transcript_file_path = os.path.join(self.transcript_dir, f"{episode_id}_transcript.txt")
        
        # In a real implementation with actual audio files:
        if os.path.exists(transcript_file_path): 
            return transcript_file_path
        
        result = self.transcription_model.transcribe(audio_file_path)
        transcript = result["text"]
        
        # For demonstration:
        print(f"The audio would be transcribed using Whisper")
        print(f"Transcript would be saved to: {transcript_file_path}")
        
        with open(transcript_file_path, 'w') as f:
            f.write(transcript)
        
        return transcript_file_path
    
    def create_vector_database(self, transcript_files):
        """
        Create a vector database from the transcripts for semantic search.
        
        Args:
            transcript_files (list): List of paths to transcript files
            
        Returns:
            object: The vector database for retrieval
        """
        print("Creating vector database from transcripts")
        
        # Load all transcripts
        documents = []
        for file_path in transcript_files:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
        
        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        db = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        
        # Persist the database
        db.persist()
        print(f"Vector database created and saved to {self.db_dir}")
        
        return db
    
    def setup_qa_system(self, vector_db):
        """
        Set up a question-answering system using the vector database.
        
        Args:
            vector_db: The vector database for retrieval
            
        Returns:
            object: The QA system
        """
        print("Setting up QA system")
        
        # Create a retriever from the vector store
        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create a prompt template
        template = """
        You are an assistant that answers questions about podcast episodes.
        Use the following context to answer the question. If you don't know the answer,
        just say you don't know. Don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain with Ollama
        llm = Ollama(model=self.model_name, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    def process_podcast(self, podcast_url, episode_id=None):
        """
        Process a podcast episode: download, transcribe, and prepare for QA.
        
        Args:
            podcast_url (str): URL of the podcast episode
            episode_id (str, optional): ID to assign to the episode
            
        Returns:
            object: The QA system for this podcast
        """
        # Download audio
        audio_path = self.download_podcast_audio(podcast_url, episode_id)
        
        # Generate transcript
        transcript_path = self.generate_transcript(audio_path)
        
        # Create vector database
        vector_db = self.create_vector_database([transcript_path])
        
        # Set up QA system
        qa_system = self.setup_qa_system(vector_db)
        
        return qa_system
    
    def process_multiple_episodes(self, podcast_urls):
        """
        Process multiple podcast episodes and create a unified QA system.
        
        Args:
            podcast_urls (list): List of podcast URLs to process
            
        Returns:
            object: The unified QA system
        """
        transcript_paths = []
        
        # Process each podcast
        for i, url in enumerate(podcast_urls):
            episode_id = f"episode_{i+1}"
            audio_path = self.download_podcast_audio(url, episode_id)
            transcript_path = self.generate_transcript(audio_path)
            transcript_paths.append(transcript_path)
        
        # Create unified vector database
        vector_db = self.create_vector_database(transcript_paths)
        
        # Set up unified QA system
        qa_system = self.setup_qa_system(vector_db)
        
        return qa_system
    
    def ask_question(self, qa_system, question):
        """
        Ask a question to the QA system.
        
        Args:
            qa_system: The QA system
            question (str): The question to ask
            
        Returns:
            str: The answer
        """
        result = qa_system({"query": question})
        return result["result"]

def main():
    parser = argparse.ArgumentParser(description="Podcast Analysis System with Ollama")
    parser.add_argument("--model", type=str, default="deepseek-r1:32b", help="Ollama model name (default: deepseek-r1:32b)")
    parser.add_argument("--urls", type=str, nargs="+", required=True, help="Podcast episode URLs")
    args = parser.parse_args()
    
    processor = PodcastProcessor(model_name=args.model)
    qa_system = processor.process_multiple_episodes(args.urls)
    
    # Interactive QA loop
    print("\nPodcast QA System Ready! Type 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == "exit":
            break
        
        answer = processor.ask_question(qa_system, question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()