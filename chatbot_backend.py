from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
# for add all messages to state
from langgraph.graph.message import add_messages
# for memory saver (sqlite)
from langgraph.checkpoint.sqlite import SqliteSaver
# for tools
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from dotenv import load_dotenv
import sqlite3
# load_dotenv()

# define tools
search = DuckDuckGoSearchRun(description="Search for information on the internet")

@tool
def get_jobs_tool(job_title: str):
    """Useful for when you need to find jobs. Input should be a job title."""
    url = "https://jobs-api14.p.rapidapi.com/v2/linkedin/search"

    querystring = {"query":job_title,"experienceLevels":"intern;entry;associate;midSenior;director","workplaceTypes":"remote;hybrid;onSite","location":"Bangladesh","datePosted":"month","employmentTypes":"contractor;fulltime;parttime;intern;temporary"}

    headers = {
        "x-rapidapi-key": "1fe1f28520msh1e8fbf05c7c6a94p1a35ecjsnd51a81fb3246",
        "x-rapidapi-host": "jobs-api14.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    return response.json()
    # print(response.json())

tools = [search, get_jobs_tool]

# define llm
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key="AIzaSyC8k0BJEofzDvH9pXmUCEsilPlSBSZmf5s")

# define llm with tools
llm_with_tools = llm.bind_tools(tools=tools)

# define state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# define functions
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    all_messages = state['messages']
    llm_response = llm_with_tools.invoke(all_messages)
    return {'messages': [llm_response]}

# connect to database
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# define memory
check_pointer = SqliteSaver(conn=conn)

# define tool node
tool_node = ToolNode(tools=tools)

# define state
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=check_pointer)

thread_id = "thread_id_1"

config = {'configurable': {"thread_id": thread_id}}


# check chatbot with tools working
# response = chatbot.invoke({"messages": [HumanMessage(content="I am a laravel developer. What jobs are available for me?")]}, config=config)

# print(response['messages'][-1].content)

# get all the thread_ids from the database checkpointer
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in check_pointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
# print(retrieve_all_threads())