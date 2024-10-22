import asyncio
import json
import os
from openai import AsyncOpenAI
from termcolor import colored

# Initialize OpenAI clients
o1_client = AsyncOpenAI()
perplexity_client = AsyncOpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

async def agent_response(messages):
    response = await o1_client.chat.completions.create(
        model="o1-mini",
        messages=messages
    )
    return response.choices[0].message.content

async def web_search(query):
    print("searching the web for: ", query)
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

def parse_agent_response(response):
    search_terms = []
    actions = []
    
    # Extract search terms
    start = response.find("<search_terms>")
    end = response.find("</search_terms>")
    if start != -1 and end != -1:
        terms = response[start+14:end].strip().split(",")
        search_terms = [term.strip() for term in terms]
    
    # Extract actions
    start = response.find("<actions>")
    end = response.find("</actions>")
    if start != -1 and end != -1:
        actions = response[start+9:end].strip().split(",")
        actions = [action.strip() for action in actions]
    
    return search_terms, actions

async def perform_searches(search_terms):
    tasks = [web_search(term) for term in search_terms]
    results = await asyncio.gather(*tasks)
    return dict(zip(search_terms, results))

def write_report(content):
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(content)

def save_iteration_data(iteration, agent_response, search_results):
    with open(f"iteration_{iteration}_agent.json", "w") as f:
        json.dump({"response": agent_response}, f, indent=2)
    
    with open(f"iteration_{iteration}_search_results.json", "w") as f:
        json.dump(search_results, f, indent=2)

async def main():
    print(colored("Starting the research AI agent...", "cyan"))
    
    research_question = input(colored("Please enter your research question: ", "green"))
    print(colored(f"Research question: {research_question}", "yellow"))
    
    messages = [
        {
            "role": "user",
            "content": f"""You are an autonomous research AI agent. Your task is to research the following question: {research_question}

            Follow these guidelines:
            1. Provide search terms within <search_terms></search_terms> tags.
            2. Specify actions within <actions></actions> tags.
            3. Wait for search results before taking the next action.
            4. Continue researching until you have enough information to write a comprehensive report.
            5. When ready, include 'write_report' in your actions to end the research phase.

            Your report should be in Markdown format. Be thorough and autonomous in your research process."""
        }
    ]
    
    iteration = 1
    while True:
        print(colored(f"\nIteration {iteration}", "magenta"))
        
        print(colored("Getting agent response...", "cyan"))
        response = await agent_response(messages)
        print(colored("Agent response received.", "cyan"))
        
        search_terms, actions = parse_agent_response(response)
        print(colored(f"Parsed search terms: {search_terms}", "yellow"))
        print(colored(f"Parsed actions: {actions}", "yellow"))
        
        save_iteration_data(iteration, response, {})
        
        if "write_report" in actions:
            print(colored("Agent has decided to write the report. Ending research phase.", "green"))
            write_report(response)
            break
        
        if search_terms:
            print(colored("Performing web searches...", "cyan"))
            search_results = await perform_searches(search_terms)
            print(colored("Web searches completed.", "cyan"))
            
            save_iteration_data(iteration, response, search_results)
            
            search_results_str = json.dumps(search_results, indent=2)
            messages.append({"role": "user", "content": f"Search results: {search_results_str}\n\nBased on these results, continue your research process."})
        
        iteration += 1

if __name__ == "__main__":
    asyncio.run(main())

