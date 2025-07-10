import argparse
import os
import shutil
from typing import List

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import wikipedia
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("🧹 Clearing database...")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents() -> List[Document]:
    """
    Loads Wikipedia articles on Joseon history in both English and Korean.
    """
    all_documents = []

    # === Load English articles ===
    wikipedia.set_lang("en")
    en_topics = wikipedia.search("Joseon", results=200)
    print(f"📚 Loading {len(en_topics)} English Wikipedia articles...")
    for topic in en_topics:
        try:
            loader = WikipediaLoader(query=topic, lang="en", load_max_docs=1, doc_content_chars_max=100000)
            docs = loader.load()
            for doc in docs:
                doc.metadata["topic"] = topic
                doc.metadata["lang"] = "en"
            all_documents.extend(docs)
            print(f"  - 🇺🇸 Loaded '{topic}'")
        except Exception as e:
            print(f"  - ❗️ Failed to load EN '{topic}': {e}")

    # === Load Korean articles ===
    wikipedia.set_lang("ko")
    ko_topics = wikipedia.search("조선", results=200)
    print(f"📚 Loading {len(ko_topics)} Korean Wikipedia articles...")
    for topic in ko_topics:
        try:
            loader = WikipediaLoader(query=topic, lang="ko", load_max_docs=1, doc_content_chars_max=100000)
            docs = loader.load()
            for doc in docs:
                doc.metadata["topic"] = topic
                doc.metadata["lang"] = "ko"
            all_documents.extend(docs)
            print(f"  - 🇰🇷 Loaded '{topic}'")
        except Exception as e:
            print(f"  - ❗️ Failed to load KO '{topic}': {e}")

    print(f"✅ Loaded {len(all_documents)} total documents.")
    return all_documents

    
def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits the loaded documents into smaller chunks for processing.
    """
    print("🔪 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks

def add_to_chroma(chunks: List[Document]):
    """
    Adds new document chunks to the Chroma vector store.
    It checks for existing chunks to avoid duplication.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    chunks_with_ids = assign_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"📦 Found {len(existing_ids)} existing documents in the DB.")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"➕ Adding {len(new_chunks)} new chunks to the DB...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        print("✅ New chunks added and database saved.")
    else:
        print("✅ No new documents to add. Database is up to date.")

def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
    """
    Assigns a unique and deterministic ID to each chunk based on its source topic and position.
    Format: "<topic>:<chunk_index>"
    """
    last_topic = None
    chunk_index = 0
    for chunk in chunks:
        topic = chunk.metadata.get("topic", "unknown")
        
        if topic == last_topic:
            chunk_index += 1
        else:
            chunk_index = 0
            last_topic = topic

        lang = chunk.metadata.get("lang", "unknown")
        chunk_id = f"{lang}:{topic}:{chunk_index}"
        chunk.metadata["id"] = chunk_id
    return chunks

def clear_database():
    """
    Deletes the Chroma database directory if it exists.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("✅ Database cleared.")

if __name__ == "__main__":
    main()