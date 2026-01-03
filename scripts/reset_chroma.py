import sys
import os
import shutil

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import Config
except ImportError:
    # Fallback if run from different context
    sys.path.append(os.getcwd())
    from config import Config

import chromadb
from chromadb.config import Settings

def reset_chroma():
    print(f"🧹 Resetting ChromaDB at: {Config.CHROMA_DB_PATH}")
    print("This will remove all existing embeddings. They must be re-generated.")
    
    success = False
    
    # Method 1: Use client.reset() if allowed
    try:
        print("Attempting to reset via ChromaDB API...")
        client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(allow_reset=True)
        )
        client.reset()
        print("✅ ChromaDB reset successfully via API.")
        success = True
    except Exception as e:
        print(f"⚠️ API reset failed: {e}")
    
    # Method 2: Delete directory if API failed or just to be sure
    if not success or os.path.exists(Config.CHROMA_DB_PATH):
        print("Attempting manual directory deletion...")
        try:
            if os.path.exists(Config.CHROMA_DB_PATH):
                shutil.rmtree(Config.CHROMA_DB_PATH)
                print("✅ ChromaDB directory deleted manually.")
            else:
                print("Directory already gone.")
            success = True
        except Exception as e:
            print(f"❌ Failed to delete directory: {e}")
            success = False
            
    if success:
        print("\n✨ Done! Restart the application to re-initialize an empty vector store.")

if __name__ == "__main__":
    reset_chroma()
