import os
import warnings
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

# Initialize the model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the CSV agent
agent = create_csv_agent(
    llm,
    "pokemon.csv",  # Change this to your CSV file
    verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True,
    handle_parsing_errors=True
)

print("CSV AI Assistant is running! Type 'exit' to stop.\n")

# Initial explanation of CSV contents
try:
    prompt = (
        "Explain the structure of the CSV file using plain language only — "
        "do not use code or assume a variable like 'df'. "
        "List the columns, their meanings, types, and any patterns you find. "
        "Then suggest example questions to explore the data."
    )
    csv_details = agent.run(prompt)
    print("AI:", csv_details)
except Exception as e:
    print(f"AI: Sorry, I couldn't analyze the CSV file. Error: {str(e)}")

print("\nHow can I help you?")

# Chat loop
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Add to memory
    memory.chat_memory.add_user_message(user_input)
    chat_history_str = "\n".join([msg.content for msg in memory.chat_memory.messages])
    full_prompt = f"{chat_history_str}\nUser: {user_input}"

    try:
        response = agent.run(full_prompt)
        memory.chat_memory.add_ai_message(response)
        print("AI:", response)
    except Exception as e:
        print("Sorry, I didn’t quite understand that. Make sure your question relates to the CSV file.")
