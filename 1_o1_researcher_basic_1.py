import asyncio
import json
import os
from typing import List, Dict
from openai import AsyncOpenAI
from termcolor import colored


from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Retrieve the token
# token = os.getenv('PERPLEXITY_API_KEY')

# # Use the token
# print(f"My token is: {token}")


# Initialize OpenAI clients
agent_client = AsyncOpenAI()
perplexity_client = AsyncOpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

async def agent_response(messages: List[Dict[str, str]]) -> str:
    print(colored("Agent is thinking...", "yellow"))
    response = await agent_client.chat.completions.create(
        model="o1-mini",
        messages=messages
    )
    return response.choices[0].message.content

async def web_search(query: str) -> str:
    print(colored(f"Searching the web for: {query}", "cyan"))
    messages = [
        {
            "role": "system",
            "content": "Perform in-depth web searches and return detailed results."
        },
        {
            "role": "user",
            "content": query
        }
    ]
    response = await perplexity_client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages
    )
    return response.choices[0].message.content

def parse_agent_response(response: str) -> Dict[str, List[str]]:
    search_terms = []
    actions = []
    
    for line in response.split('\n'):
        if line.startswith('<search_terms>') and line.endswith('</search_terms>'):
            terms = line[14:-15].strip()
            search_terms.extend([term.strip() for term in terms.split(',')])
        elif line.startswith('<action>') and line.endswith('</action>'):
            actions.append(line[8:-9].strip())
    
    return {"search_terms": search_terms, "actions": actions}

async def perform_searches(search_terms: List[str]) -> List[str]:
    tasks = [web_search(term) for term in search_terms]
    return await asyncio.gather(*tasks)

def save_to_json(data: Dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def write_report(content: str):
    with open("research_report.txt", "w", encoding="utf-8") as f:
        f.write(content)

async def research_loop(initial_question: str):
    messages = [
        {
            "role": "user",
            "content": (
                "You are an autonomous research AI agent. Your task is to research the following question: "
                f"{initial_question}\n\n"
                "Provide search terms within <search_terms></search_terms> tags and actions within <action></action> tags. "
                "Continue researching until you have enough information to write a comprehensive report. "
                "When you're ready to write the report, use the <action>write_report</action> tag."
            )
        }
    ]
    
    iteration = 1
    while True:
        print(colored(f"\nIteration {iteration}", "magenta"))
        
        response = await agent_response(messages)
        parsed_response = parse_agent_response(response)
        
        save_to_json({"agent_response": response, "parsed_response": parsed_response}, f"iteration_{iteration}_agent.json")
        
        if "write_report" in parsed_response["actions"]:
            print(colored("Agent is ready to write the report.", "green"))
            report_content = await agent_response(messages + [{"role": "user", "content": "Please write a comprehensive report based on your research."}])
            write_report(report_content)
            print(colored("Report has been written to research_report.txt", "green"))
            break
        
        search_results = await perform_searches(parsed_response["search_terms"])
        save_to_json({"search_results": search_results}, f"iteration_{iteration}_search_results.json")
        
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Search results:\n{json.dumps(search_results)}\n\nContinue your research based on these results."})
        
        iteration += 1

async def main():
    print(colored("Welcome to the Research AI Agent!", "blue"))
    initial_question = input(colored("Please enter your research question: ", "green"))
    await research_loop(initial_question)

if __name__ == "__main__":
    asyncio.run(main())

