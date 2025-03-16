# app.py
from flask import Flask, render_template, request, jsonify, session
import requests
import os
import re
import json
from datetime import datetime
import whisper
import feedparser
import uuid
import shutil
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
import soundfile as sf
import base64
import io

from audio.generator import load_csm_1b
import torch


app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'podcast_data/audio'
app.config['TRANSCRIPT_FOLDER'] = 'podcast_data/transcripts'
app.config['DB_FOLDER'] = 'podcast_data/vectordb'

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['TRANSCRIPT_FOLDER'], app.config['DB_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Global variables
OLLAMA_MODEL = "qwen2.5:32b" #qwq:latest, qwen2.5:32b, deepseek-r1:32b, llama3.3:latest, nemotron:latest
PODCAST_RSS_URL = "https://feeds.megaphone.fm/empirepodcast"
CONVERSATION_MEMORY = {}
WHISPER_MODEL = "small" #https://github.com/openai/whisper/blob/main/README.md
TRANSCRIPT_SIZE = 100000
AUDIO_LENGTH=10_000


def remove_think_tags(text):
  """Removes <think> and </think> tags and their content from a string."""
  pattern = r"<think>.*?</think>"
  return re.sub(pattern, "", text, flags=re.DOTALL)

def check_ollama_running():
    """Verify that Ollama is running and the model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        available_models = [model["name"] for model in response.json()["models"]]
        
        if OLLAMA_MODEL not in available_models:
            return False, f"Model '{OLLAMA_MODEL}' not found in Ollama. Available models: {available_models}"
        return True, "Ollama is running with the required model"
    except requests.exceptions.ConnectionError:
        return False, "Ollama service is not running. Start Ollama before using this application."

def get_podcast_episodes():
    """Fetch episodes from the podcast RSS feed"""
    feed = feedparser.parse(PODCAST_RSS_URL)
    episodes = []
    
    for entry in feed.entries:
        episode_id = entry.id if hasattr(entry, 'id') else str(uuid.uuid4())
        db_path = os.path.join(app.config['DB_FOLDER'], episode_id)

        episode = {
            'title': entry.title,
            'date': entry.published,
            'summary': entry.summary if hasattr(entry, 'summary') else '',
            'id': episode_id,
            'processed': True if os.path.exists(db_path) else False,
        }
        
        # Find the audio file URL
        for link in entry.links:
            if link.get('type', '').startswith('audio/'):
                episode['audio_url'] = link.href
                break
        
        if 'audio_url' not in episode:
            for enclosure in entry.get('enclosures', []):
                if enclosure.get('type', '').startswith('audio/'):
                    episode['audio_url'] = enclosure.href
                    break
        
        episodes.append(episode)

    for episode in episodes:
        episode['datetime'] = parse_date(episode['date'])

    episodes = sorted(episodes, key=lambda x: x['datetime'] if x['datetime'] else datetime.min, reverse=False) # Sort ascending.
    
    return episodes
    

# Convert 'date' strings to datetime objects and sort
def parse_date(date_string):
    """Parses a date string into a datetime object."""
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%a, %d %b %Y',
        '%d %b %Y %H:%M:%S %z',
        '%d %b %Y %H:%M:%S',
        '%Y-%m-%d',
        '%a, %d %b %Y %H:%M:%S %Z'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            pass
    return None #Return none if no format matches.


def download_podcast(episode_id, audio_url):
    """Download podcast audio file"""
    audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{episode_id}.mp3")
    
    # Check if the file already exists
    if os.path.exists(audio_file_path): 
        return audio_file_path

    # Download the audio file    
    response = requests.get(audio_url)
    with open(audio_file_path, 'wb') as f:
        f.write(response.content)
    
    
    return audio_file_path

def generate_transcript(episode_id, audio_file_path):
    """Generate transcript from audio file"""
    transcript_file_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{episode_id}_transcript.txt")
    
    if os.path.exists(transcript_file_path): 
        return transcript_file_path
        
    # Transcribe the audio file
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_file_path)
    transcript = result["text"]    
   
   # Write the transcript to a file
    with open(transcript_file_path, 'w') as f:
        f.write(transcript)
    
    return transcript_file_path

def create_vector_db(episode_id, transcript_path):
    """Create vector database from transcript"""
    db_path = os.path.join(app.config['DB_FOLDER'], episode_id)

    if os.path.exists(db_path): 
        return db_path

    # Load transcript
    loader = TextLoader(transcript_path)
    documents = loader.load()
    
    # Split texts into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    # Persist the database
    # db.persist()
    
    return db_path

def setup_qa_system(db_path, session_id):
    """Set up QA system for the episode"""
    # Load the vector store
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Create a retriever
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    # Create memory for conversation
    if session_id not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",  
            return_messages=True
        )
    
    # Create a prompt template
    template = """
    You are an assistant that answers questions about podcast episodes.
    Use the following context to answer the question. If you don't know the answer,
    just say you don't know. Don't try to make up an answer.
    
    Context: {context}
    
    Chat History: {chat_history}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create the QA chain with properly configured return_source_documents
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,  
        chain_type_kwargs={
            "prompt": prompt,
            "memory": CONVERSATION_MEMORY[session_id]
        }
    )
    
    return qa_chain

def process_episode(episode_id, audio_url):
    """Process a podcast episode"""
    # Download the audio
    audio_path = download_podcast(episode_id, audio_url)
    
    # Generate transcript
    transcript_path = generate_transcript(episode_id, audio_path)
    
    # Create vector database
    db_path = create_vector_db(episode_id, transcript_path)
    
    return {
        "status": "success",
        "message": "Episode processed successfully",
        "db_path": db_path
    }

def get_processed_episodes(folder_path):
    """
    Looks through a folder and returns a list of dictionaries, where each dictionary
    represents a processed episode.

    Args:
        folder_path (str): The path to the folder containing episode folders.

    Returns:
        list: A list of dictionaries, where each dictionary contains 'id' and 'db_path'.
    """
    # Get podcast episodes
    episodes = get_podcast_episodes()

    processed_episodes = []
    for episode_id in os.listdir(folder_path):
        db_path = os.path.join(folder_path, episode_id)
        # Check if it's a directory
        if os.path.isdir(db_path):
            # Check if the db file exists.
            if os.path.exists(db_path):
                # find the episode title
                episode_title = None    
                for episode in episodes:
                    if episode['id'] == episode_id:
                        episode_title = episode['title']
                        break # break once found                            
                
                processed_episodes.append({
                    'id': episode_id,
                    'db_path': db_path,
                    'title': episode_title,
                })
            else:
                print(f"Warning: Database file not found for episode {episode_id} at {db_path}") #optional warning message.
        else:
          print(f"Warning: {episode_id} is not a directory. Skipping") #optional warning message.

    return processed_episodes

def get_episode_name_by_id(episode_id):
    """Find the episode name for a given episode ID."""
    episodes = get_podcast_episodes()
    for episode in episodes:
        if episode['id'] == episode_id:
            return episode['title']
    return None
    
@app.route('/')
def index():
    """Render the home page"""
    # Check if Ollama is running
    ollama_status, message = check_ollama_running()
    if not ollama_status:
        return render_template('error.html', message=message)
    
    # Get podcast episodes
    episodes = get_podcast_episodes()
    
    return render_template('index.html', episodes=episodes)

@app.route('/process', methods=['POST'])
def process():
    """Process a podcast episode"""
    data = request.json
    episode_id = data.get('id')
    audio_url = data.get('audio_url')
    
    if not episode_id or not audio_url:
        return jsonify({"status": "error", "message": "Missing episode ID or audio URL"})
    
    result = process_episode(episode_id, audio_url)
    
    # Save the processed episode to the session
    if 'processed_episodes' not in session:
        session['processed_episodes'] = []
    
    session['processed_episodes'].append({
        'id': episode_id,
        'db_path': result['db_path']
    })
    
    return jsonify(result)

@app.route('/chat')
def chat():
    """Render the chat page"""

    # Save the processed episode to the session
    if 'processed_episodes' not in session:
        session['processed_episodes'] = []
    
    session['processed_episodes'] = get_processed_episodes(app.config['DB_FOLDER'])

    # Check if any episodes have been processed
    if 'processed_episodes' not in session or not session['processed_episodes'] or len(session['processed_episodes']) == 0:
        return render_template('error.html', message="No episodes have been processed yet")
    
    # Generate a unique session ID for this chat
    if 'chat_session_id' not in session:
        session['chat_session_id'] = str(uuid.uuid4())
    
    return render_template('chat.html', episodes=session['processed_episodes'])

@app.route('/ask', methods=['POST'])
def ask():
    """Handle chat questions"""
    data = request.json
    question = data.get('question')
    episode_id = data.get('episode_id')
    
    if not question or not episode_id:
        return jsonify({"status": "error", "message": "Missing question or episode ID"})
    
    # Find the DB path for this episode
    db_path = None
    for ep in session.get('processed_episodes', []):
        if ep['id'] == episode_id:
            db_path = ep['db_path']
            break
    
    if not db_path:
        return jsonify({"status": "error", "message": "Episode not found"})
    
    # Set up QA system
    qa_chain = setup_qa_system(db_path, session['chat_session_id'])
    
    # Get the answer - Fix this line
    response = qa_chain.invoke({"query": question})
    
    return jsonify({
        "status": "success",
        "answer": remove_think_tags(response.get("answer", response.get("result", "No answer provided")))
    })

# Add these routes to app.py

@app.route('/summary')
def summary():
    """Render the summary page"""
    # Check if Ollama is running
    ollama_status, message = check_ollama_running()
    if not ollama_status:
        return render_template('error.html', message=message)
    
    # Get all processed episodes
    processed_episodes = get_processed_episodes(app.config['DB_FOLDER'])
    
    # Check if any episodes have been processed
    if not processed_episodes or len(processed_episodes) == 0:
        return render_template('error.html', message="No episodes have been processed yet. Please process episodes first.")
    
    return render_template('summary.html', episodes=processed_episodes)

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    """Generate a summary and eye-opening highlights from an episode"""
    data = request.json
    episode_id = data.get('episode_id')
    include_highlights = data.get('include_tidbits', True)  # Reusing the existing checkbox
    summary_length = data.get('summary_length', 'medium')
    
    if not episode_id:
        return jsonify({"status": "error", "message": "Missing episode ID"})
    
    # Find the DB path for this episode
    db_path = None
    episode_title = None
    processed_episodes = get_processed_episodes(app.config['DB_FOLDER'])
    
    for ep in processed_episodes:
        if ep['id'] == episode_id:
            db_path = ep['db_path']
            episode_title = ep['title']
            break
    
    if not db_path:
        return jsonify({"status": "error", "message": "Episode not found"})
    
    # Get the transcript path
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{episode_id}_transcript.txt")
    if not os.path.exists(transcript_path):
        return jsonify({"status": "error", "message": "Transcript not found for this episode"})
    
    # Load the transcript
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    
    # Set up the LLM
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)
    
    # Create the summary prompt based on length
    length_description = {
        'short': 'a concise 1-2 paragraph summary',
        'medium': 'a comprehensive 3-4 paragraph summary',
        'long': 'a detailed 5+ paragraph summary'
    }
    
    # Generate the summary
    summary_prompt = f"""
    You are an expert podcast summarizer. Provide {length_description[summary_length]} of the following podcast transcript. 
    Focus on the main topics discussed, key insights, and most important information.
    
    Episode Title: {episode_title}
    
    Transcript:
    {transcript[:TRANSCRIPT_SIZE]}  # Limit transcript length to avoid context window issues
    
    Summary:
    """
    
    summary_result = llm.invoke(summary_prompt)
    
    result = {
        "status": "success",
        "summary": remove_think_tags(summary_result),
        "episode_title": episode_title
    }
    
    # Generate eye-opening highlights if requested
    if include_highlights:
        highlights_prompt = f"""
        Extract 6-8 eye-opening highlights from this podcast transcript. These should be the most surprising, 
        thought-provoking, or perspective-changing moments from the episode.
        
        For each highlight, provide:
        1. An attention-grabbing headline (5-10 words)
        2. A concise explanation of why this insight is eye-opening or perspective-changing (1-2 sentences)
        
        Format each highlight as a JSON object with keys "headline" and "highlight".
        Enclose the entire set of highlights within a JSON array.

        Example:
        [
          {{
            "headline": "Hidden Cost of Free Services",
            "highlight": "Most 'free' digital services make money by collecting and selling user data, making users the product rather than the customer."
          }},
          {{
            "headline": "Sleep's Critical Role in Decision Making",
            "highlight": "Just one night of poor sleep reduces decision-making ability by up to 40%, comparable to being legally intoxicated."
          }}
        ]
        
        Episode Title: {episode_title}
        
        Transcript:
        {transcript[:TRANSCRIPT_SIZE]}
        
        Eye-Opening Highlights (formatted as JSON array):
        """
        
        highlights_response = llm.invoke(highlights_prompt)
        
        # Extract JSON from the response
        try:
            # Try to find JSON array in the response

            
            # Look for content that appears to be JSON
            highlights_response = remove_think_tags(highlights_response)
            json_match = re.search(r'\[\s*{.+}\s*\]', highlights_response, re.DOTALL)
            if json_match:
                highlights_text = json_match.group(0)
                highlights = json.loads(highlights_text)
            else:
                # Fallback if the response isn't properly formatted JSON
                # Create structured highlights manually
                lines = highlights_response.split('\n')
                highlights = []
                current_highlight = {}
                
                for line in lines:
                    if "Headline:" in line or "headline:" in line:
                        if current_highlight and "headline" in current_highlight:
                            highlights.append(current_highlight)
                            current_highlight = {}
                        current_highlight["headline"] = line.split(":", 1)[1].strip()
                    elif "Highlight:" in line or "highlight:" in line:
                        current_highlight["highlight"] = line.split(":", 1)[1].strip()
                
                if current_highlight and "headline" in current_highlight:
                    highlights.append(current_highlight)
            
            result["highlights"] = highlights
        except Exception as e:
            print(f"Error parsing highlights: {e}")
            result["highlights"] = []
    
    return jsonify(result)

@app.route('/delete_episode', methods=['POST'])
def delete_episode():
    """Delete a processed episode including audio file, transcript, and vector database"""
    data = request.json
    episode_id = data.get('id')
    
    if not episode_id:
        return jsonify({"status": "error", "message": "Missing episode ID"})
    
    try:
        # Delete audio file
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{episode_id}.mp3")
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        
        # Delete transcript file
        transcript_file_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{episode_id}_transcript.txt")
        if os.path.exists(transcript_file_path):
            os.remove(transcript_file_path)
        
        # Delete vector database folder
        db_path = os.path.join(app.config['DB_FOLDER'], episode_id)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        # Remove from session if present
        if 'processed_episodes' in session:
            session['processed_episodes'] = [ep for ep in session['processed_episodes'] if ep['id'] != episode_id]
        
        # Clean up any conversation memory for this episode
        session_keys_to_remove = []
        for session_id in list(CONVERSATION_MEMORY.keys()):
            if session_id.startswith(episode_id):
                session_keys_to_remove.append(session_id)
        
        for key in session_keys_to_remove:
            if key in CONVERSATION_MEMORY:
                del CONVERSATION_MEMORY[key]
        
        return jsonify({
            "status": "success", 
            "message": "Episode deleted successfully"
        })
    
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Error deleting episode: {str(e)}"
        })
    
@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech using the sesame/csm-1b model from Hugging Face"""
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"status": "error", "message": "No text provided"})
    
    try:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        generator = load_csm_1b(device=device)

        audio = generator.generate(
            text= text,
            speaker=0,
            context=[],
            max_audio_length_ms=AUDIO_LENGTH,
        )

        # Convert to audio
        audio_data = audio.cpu().numpy()        

        # Save to in-memory file
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data.squeeze(), generator.sample_rate, format='WAV')
        audio_buffer.seek(0)
        
        # Encode as base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "audio": audio_base64
        })
    except Exception as e:
        print(e)
        return jsonify({
            "status": "error", 
            "message": f"Error generating speech: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True)