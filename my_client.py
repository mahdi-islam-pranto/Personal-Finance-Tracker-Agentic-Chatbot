import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage 
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json

load_dotenv()

SERVERS = {
    "expense": {
        "transport": "streamable_http",  # if this fails, try "sse"
        "url": "https://finance-tracker-mcp.fastmcp.app/mcp"
    },
}

async def main():
    
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    print("Available tools:", named_tools.keys())

    llm = ChatOpenAI(model="gpt-5")
    # gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)

    # prompt = "add a expense of tk 300tk for transport on 12th november 2025. I went to office on a uber"
    prompt = "Show me my all expenses on novenber 2025"
    prompt2 = "Show me my all expenses today"

    # prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """ You are a Personal Finance Assistant chatbot.  
Your job is to help the user track expenses, view past spending, and summarize spending patterns.  
You MUST use the available MCP tools when appropriate.

====================
YOUR RULES
====================
1. When the user wants to add an expense:
   - ALWAYS call the `add_expense` tool.
   - Extract: date, amount, category, subcategory (if given), note (optional).
   - If user hasn't provided a date, consider it today.
   - If any important field is missing, ask one clarifying question.

2. When the user wants to see past expenses:
   - ALWAYS call the `list_expenses` tool.
   - Determine the date range from user input.
   - If no date range is provided, ask: 
     “From which date to which date should I show your expenses?”

3. When the user wants summaries or totals:
   - ALWAYS call the `summarize` tool.
   - Extract: start_date, end_date, category (optional)
   - If needed, ask the user for missing dates.

4. NEVER hallucinate data.
   You only know what comes from the database (via tool results).

5. ALWAYS return results in a friendly and organized way.

====================
OUTPUT FORMAT
====================
When using a tool → respond ONLY with a valid tool call.

Otherwise → respond conversationally.

====================
CONTEXTUAL GUIDELINES
====================
- Infer category and subcategory naturally from descriptions.
- If the user uses words like “today”, “yesterday”, “last month”, convert to ISO date (YYYY-MM-DD).
- If user gives a vague request like “show me everything,” ask which dates.
- Keep responses concise and helpful.
 """),
        ("user", """ I want to track my expenses. Use the expense tool based on the my input. {prompt} """),
    ])    
    
    response = await llm_with_tools.ainvoke(prompt_template.format_messages(prompt=prompt))

    if not getattr(response, "tool_calls", None):
        print("\nLLM Reply:", response.content)
        return


    tool_messages = []
    for tc in response.tool_calls:
        selected_tool = tc["name"]
        selected_tool_args = tc.get("args") or {}
        selected_tool_id = tc["id"]

        result = await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_messages.append(ToolMessage(tool_call_id=selected_tool_id, content=json.dumps(result)))
        

    final_response = await llm_with_tools.ainvoke([prompt, response, *tool_messages])
    print(f"Final response: {final_response.content}")


if __name__ == '__main__':
    asyncio.run(main())