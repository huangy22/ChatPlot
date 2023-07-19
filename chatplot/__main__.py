from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from agent import AGENT_EXECUTOR
user_question = input("Please input your question:")
AGENT_EXECUTOR.run(
    user_question,
)
