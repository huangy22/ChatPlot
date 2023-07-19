from typing import List
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool
from langchain.schema import HumanMessage, SystemMessage

from tools import TOOLS

CHAT_PROMPT_TEMPLATE = """
    You are an assistant that only helps with visualization tasks, such as producing figure or chart for certain data.
    Have a conversation with human, figure out the data you need to use.
    Then Ask human the type of chart you need to plot with python.
    Then provide a python code that produce the chart.
    Try to fix all bugs in the code.
    If you can not fix a bug after trying twice, show human the code and ask for help.
    Improve the code by asking human for feed back until human is satisfied.
    If you encounter some error, identify which line of codes cause the error, and analyze the reason by printing out information that can be
    relevant to the error message. For example, you can print out the columns of data if you encounter
    a key error.
    You can search for suggestions and tips, but you should always use a key word as specific as possible.
    For example, if you want to produce a bar chart with python, and needs to make it looks better,
    search for "python beautiful bar chart tips".
    As another example, you can use the entire error message as key word to search for solution.

    You have access to the following tools:
    {tools}

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do, and if human said they are satisfied.
    Action: the action to take, should be one of [{tool_names}].
    Action Input: the input to the action.
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: Since human is satisfied, I now know the final answer
    Final Answer: the final answer to the original input question. Must provide a python code template, as well as the result of executing the code.

    Begin! Remember to give detailed, informative answers.

    Previous conversation history:
    {history}

    New question: {input}
    {agent_scratchpad}
"""


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)

        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


CHAT_PROMPT = CustomPromptTemplate(
    template=CHAT_PROMPT_TEMPLATE,
    tools=TOOLS,
    input_variables=["input", "intermediate_steps", "history"]
)
