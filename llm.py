import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Create the LLM
llm = ChatOllama(model="llama3.1:8b")

# Create the Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")