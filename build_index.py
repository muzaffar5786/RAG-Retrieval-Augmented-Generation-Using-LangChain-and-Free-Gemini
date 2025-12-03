import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "medical_transcriptions.csv"   # <-- your dataset file (CSV)
SAVE_PATH = "medical_faiss_store"

def build_faiss():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["transcription"])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = []
    metadatas = []

    for idx, row in df.iterrows():
        chunks = splitter.split_text(str(row["transcription"]))
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "row_id": int(idx),
                "medical_specialty": row["medical_specialty"],
                "sample_name": row["sample_name"]
            })

    print("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    print("Building FAISS index...")
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    db.save_local(SAVE_PATH)

    print("FAISS index built and saved successfully!")

if __name__ == "__main__":
    build_faiss()
