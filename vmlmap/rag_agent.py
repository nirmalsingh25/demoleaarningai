from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from vmlmap.config import Config


def build_vml_retriever_tool(retriever):

    @tool
    def vml_retriever_tool(query: str) -> str:
        """
        Retrieve top product reviews related to the user query.
        """
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    return vml_retriever_tool


class RAGAgentBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = init_chat_model(Config.RAG_MODEL)

    def build_agent(self):

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        vml_tool = build_vml_retriever_tool(retriever)


        agent = create_agent(
            model=self.model,
            tools=[vml_tool],
            system_prompt="""
            
            You're an e-commerce bot answering product-related queries 
                    based on reviews and titles.
                    
                    And To find the answers always use 
                    vml_retriever_tool
                    
                    if you do not know an 
                    answer politely say that 
                    i dont know the answer please 
                    contact our customer care +1 437-973-****.
            
            """,
            checkpointer=InMemorySaver(),
            middleware=[
                SummarizationMiddleware(
                    model=self.model,
                    trigger=("messages", 10),
                    keep=("messages", 4),
                )
            ],
        )

        return agent
