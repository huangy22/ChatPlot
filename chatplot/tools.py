from langchain import OpenAI, LLMChain
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.tools import HumanInputRun
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools.python.tool import PythonREPLTool

REPL_TOOL = Tool(
    name="python_repl",
    description="""
    A Python shell. Use this to execute python commands. Input should be a valid python command.
    If you want to see the output of a value, you should print it out with `print(...)`.",
    """,
    func=PythonREPLTool().run,
)

GOOGLE_SERP = Tool(
    name="search",
    func=GoogleSerperAPIWrapper().run,
    description="Useful for when you need to ask with search",
)

HUMAN_TOOL = Tool(
    name="human",
    func=HumanInputRun().run,
    description="""
    Useful for when you need to ask for human guidance.
    Use this tool more often if the question is about data or type of chart.
    """
)

LLM_TOOL = Tool(
    name='Language Model',
    func=LLMChain(
        llm=OpenAI(temperature=0),
        prompt=PromptTemplate(
            input_variables=["query"],
            template="{query}",
        ),
    ).run,
    description='Use this tool for general purpose queries and logic',
)

# Changing the description of tools can influence the agent's priority of using them
TOOLS = [
    HUMAN_TOOL, REPL_TOOL, LLM_TOOL, GOOGLE_SERP
]
