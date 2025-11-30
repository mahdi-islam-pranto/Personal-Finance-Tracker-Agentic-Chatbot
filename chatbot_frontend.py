import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
import uuid

###################### Utility functions ######################

# generate uuid for thread_id
def generate_tread_id():
    thread_id = uuid.uuid4()
    return thread_id

# reset chat history
def reset_chat():
    thread_id = generate_tread_id()
    st.session_state["thread_id"] = thread_id
    add_thread_id_to_list(st.session_state["thread_id"])
    st.session_state["message_history"] = []
    
# add thread_ id to thread_id_list
def add_thread_id_to_list(thread_id):
    # check if thread_id is already in the list
    if thread_id not in st.session_state['thread_id_list']:
        st.session_state['thread_id_list'].append(thread_id)

# load all conversations from a thread_id to message_history
def load_conversation_from_thread_id(tread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])



###################### Session state setup ######################
# initialize message history in session state
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# initialize thread_id in session state
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_tread_id()

# initialize a list of all thread_ids in session state
if "thread_id_list" not in st.session_state:
    st.session_state["thread_id_list"] = retrieve_all_threads()

add_thread_id_to_list(st.session_state['thread_id'])

# sample message history
# st.session_state["message_history"] = [
#     {"role": "user", "content": "Hello"},
#     {"role": "assistant", "content": "Hello, how can I help you?"},
# ]

##################### Sidebar UI ######################

# sidebar UI
st.sidebar.title("Your Conversations")
if st.sidebar.button("New Conversation", type="primary"):
    reset_chat()

# show all thread_ids in the sidebar 
for thread_id in st.session_state["thread_id_list"][::-1]:
    # show a part of message to show as a button
    messages = load_conversation_from_thread_id(thread_id)
    # get first message from messages
    first_message = messages[0].content if messages else "New Chat"
    if len(first_message) > 50:
        first_message = first_message[:50] + "..."

    if st.sidebar.button(first_message):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation_from_thread_id(thread_id)

        # convert messages to the format that streamlit can understand
        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        
    

##################### Main UI ######################
st.title("Ask me anything")
# get user input
user_input = st.chat_input("Type here: ")

# show message history in the chat interface
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# show user input in the chat interface
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message('user'):
        st.text(user_input)

    thread_id = st.session_state["thread_id"]

    CONFIG = {'configurable': {"thread_id": thread_id}}

    # get chatbot response
    # response = chatbot.invoke({"messages": [SystemMessage(content="You are a helpful chatbot."),
    #                               HumanMessage(content=user_input)]}, config=config )
    # ai_response = response['messages'][-1].content

    
    # show chatbot response in the chat interface
    with st.chat_message('assistant'):

        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

  
        ai_response = st.write_stream(
            ai_only_stream()

        )

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )


    # put ai response in the message history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_response})