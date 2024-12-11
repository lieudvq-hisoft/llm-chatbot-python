import streamlit as st
from llm import llm
from graph import graph

from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True
)