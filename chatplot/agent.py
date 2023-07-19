import re
from typing import Union
from langchain import OpenAI, LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory

from tools import TOOLS
from prompts import CHAT_PROMPT

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

AGENT = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=OpenAI(temperature=0), prompt=CHAT_PROMPT),
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=[tool.name for tool in TOOLS]
)

AGENT_EXECUTOR = AgentExecutor.from_agent_and_tools(
    agent=AGENT,
    tools=TOOLS,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="history"),
    #memory=ConversationBufferWindowMemory(k=2),
    #memory=ConversationSummaryMemory(llm=OpenAI(temperature=0)),
    max_iterations=100,
)
