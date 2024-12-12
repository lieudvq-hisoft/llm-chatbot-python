from llm import llm, embeddings
from graph import graph

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from pydantic import BaseModel, Field
from langchain_core.runnables import  RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import PydanticOutputParser

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

entity_chain = llm | PydanticOutputParser(pydantic_object=Entities)

def extract_entities(question: str) -> Entities:
    try:
        full_prompt = f"""Extract entities from the following text. 
        If no specific entities are found, return an empty list.
        Text: {question}
        Entities (comma-separated):"""
        response = llm.invoke(full_prompt).content
        entities = [e.strip() for e in response.split(',') if e.strip()]
        
        return Entities(names=entities)
    
    except Exception as e:
        print(f"Lỗi trích xuất thực thể: {e}")
        return Entities(names=[])

# Fulltext index query
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = extract_entities(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    final_data = f"""Graph data:
{graph_data}
vector data:
{"#Document ". join(vector_data)}
    """
    return final_data

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

vector_retriever = vector_index.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
        {
            "context": full_retriever,
            "question": RunnablePassthrough(),
            "chat_history": memory.load_memory_variables

        }
    | prompt
    | llm
    | StrOutputParser()
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    memory.chat_memory.add_user_message(user_input)
    response = chain.invoke(input=user_input)
    memory.chat_memory.add_ai_message(response)
    return response