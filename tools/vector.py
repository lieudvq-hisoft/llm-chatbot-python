import streamlit as st
from llm import llm, embeddings
from graph import graph

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate


neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    graph=graph,                             # (2)
    index_name="vector",                 # (3)
    node_label="Document",                      # (4)
    text_node_property="text",               # (5)
    embedding_node_property="embedding", # (6)
    retrieval_query="""
    RETURN
        node.text AS text,
        score,
        {
            id: node.id,
            labels: labels(node),
            outgoing_relations: [
                (node)-[r]->(neighbor) | { 
                    relationType: type(r), 
                    neighborId: neighbor.id, 
                    neighborLabels: labels(neighbor) 
                }
            ],
            incoming_relations: [
                (neighbor2)-[r2]->(node) | { 
                    relationType: type(r2), 
                    neighborId: neighbor2.id, 
                    neighborLabels: labels(neighbor2) 
                }
            ]
        } AS metadata
    """
)

retriever = neo4jvector.as_retriever()

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

def get_flutter_text(input):
    return plot_retriever.invoke({"input": input})