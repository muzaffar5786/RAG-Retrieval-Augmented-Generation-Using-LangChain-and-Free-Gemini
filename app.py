import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


st.title("Medical RAG Assistant")


@st.cache_resource
def load_faiss():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    if not os.path.exists("medical_faiss_store"):
        st.warning("Vector store not found. Building FAISS index now (first run only)...")
        os.system("python build_index.py")

    return FAISS.load_local(
        "medical_faiss_store",
        embeddings,
        allow_dangerous_deserialization=True
    )


db = load_faiss()
retriever = db.as_retriever(search_kwargs={"k": 4})


llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.1
)


from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

query = st.text_input("Ask your medical question:")

if st.button("Search"):
    res = qa_chain.invoke({"query": query})
    st.write(res["result"])

    st.subheader("Sources")
    for doc in res["source_documents"]:
        st.write(doc.page_content)
