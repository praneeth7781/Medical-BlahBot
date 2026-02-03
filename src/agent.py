# from phoenix.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor

# tracer_provider = register(project_name="medical-rag-agent")
# LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
# from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.prompts import GRADE_PROMPT, SYSTEM_PROMPT, ANSWER_PROMPT, REWRITE_PROMPT
from src.tools import search_medical_reference
from pydantic import BaseModel, Field
from typing import Literal, AsyncGenerator
import dotenv
from langchain_groq import ChatGroq

dotenv.load_dotenv()

MODEL_NAME = "gpt-oss:20b"
MODEL_BASE_URL = "http://localhost:11434"
MAX_RETRIES = 2

class AgentState(MessagesState):
    retry_count: int

# llm = ChatOllama(
#     model=MODEL_NAME,
#     base_url=MODEL_BASE_URL,
#     temperature=0.2
# )

# grader_llm = ChatOllama(
#     model=MODEL_NAME,
#     base_url=MODEL_BASE_URL,
#     temperature=0.0
# )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)

grader_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0
)

class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="'yes' if the documents are relevant to the question, 'no' otherwise"
    )

def agent_node(state: AgentState) -> dict:
    """
    decide whethr to search or respond directly.
    For medical questions, it should always call the search tool.
    For greetings/clarifications, it may respond directly.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.bind_tools([search_medical_reference]).invoke(messages)
    return {"messages": [response]}


def generate_answer_node(state: AgentState) -> dict:
    """
    Generate the final answer using retrieved context.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    messages.append(HumanMessage(content=ANSWER_PROMPT))
    
    response = llm.invoke(messages)
    return {"messages": [response]}


def rewrite_question_node(state: AgentState) -> dict:
    """
    Rewrite the question to improve retrieval results.
    """
    original_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            original_question = msg.content
            break
    
    prompt = REWRITE_PROMPT.format(question=original_question)
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip()
    
    new_message = HumanMessage(content=f"Search for: {rewritten}")
    
    return {
        "messages": [new_message],
        "retry_count": state.get("retry_count", 0) + 1
    }

def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """if agent wants to call tool, route to tool nde"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def route_after_tools(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    """
    grade retrieved documents and decide next step.
    If retry limit reached, proceed to generate answer regardless.
    """
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "generate_answer"
    
    question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            question = msg.content
            # break
    
    context = state["messages"][-1].content
    
    # Grade the documents
    prompt = GRADE_PROMPT.format(question=question, context=context)
    
    try:
        response = grader_llm.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        
        if response.binary_score.lower() == "yes":
            return "generate_answer"
        return "rewrite_question"
    
    except Exception:
        # On grading failure, proceed to generate answer
        return "generate_answer"

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode([search_medical_reference]))
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    
    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_conditional_edges("tools", route_after_tools)
    graph.add_edge("generate_answer", END)
    graph.add_edge("rewrite_question", "agent")
    
    return graph.compile()

class Graph:

    def __init__(self):
        self.app = build_graph()
        self.messages: list = []
    
    async def stream(self, user_message: str) -> AsyncGenerator[str, None]:
        self.messages.append(HumanMessage(content=user_message))
        
        state = {
            "messages": self.messages.copy(),
            "retry_count": 0
        }
        
        full_response = ""
        agent_buffer = ""
        used_tools = False
        
        try:
            async for event in self.app.astream_events(state, version="v2"):
                if not isinstance(event, dict):
                    continue
                
                event_type = event.get("event", "")
                metadata = event.get("metadata", {})
                node_name = metadata.get("langgraph_node", "")
                
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        
                        if node_name == "generate_answer":
                            full_response += chunk.content
                            yield chunk.content
                        
                        elif node_name == "agent":
                            agent_buffer += chunk.content
                
                elif event_type == "on_chain_end" and node_name == "agent":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        messages = output.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                used_tools = True
            
            if not used_tools and agent_buffer and not full_response:
                full_response = agent_buffer
                yield agent_buffer
            
            if full_response:
                self.messages.append(AIMessage(content=full_response))
        
        except Exception as e:
            error_message = f"I encountered an error while processing your request. Please try again. {str(e)}"
            self.messages.append(AIMessage(content=error_message))
            yield error_message
    
    def reset(self) -> None:
        self.messages = []
    
    def get_history(self) -> list:
        return self.messages.copy()
