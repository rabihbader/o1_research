import asyncio
import json
from openai import AsyncOpenAI
import os
from termcolor import colored

agent_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
search_client = AsyncOpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")

def parse_agent_response(response):
    search_terms = []
    actions = []
    report = None
    
    # Extract search terms
    start_tag = "<search_terms>"
    end_tag = "</search_terms>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)
    if start_index != -1 and end_index != -1:
        search_terms_str = response[start_index + len(start_tag):end_index].strip()
        search_terms = [term.strip() for term in search_terms_str.split(",")]
    
    # Extract actions  
    start_tag = "<actions>"
    end_tag = "</actions>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)
    if start_index != -1 and end_index != -1:
        actions_str = response[start_index + len(start_tag):end_index].strip()
        actions = [action.strip() for action in actions_str.split(",")]
    
    # Extract report
    start_tag = "```markdown"
    end_tag = "```"  
    start_index = response.find(start_tag)
    end_index = response.find(end_tag, start_index + len(start_tag))
    if start_index != -1 and end_index != -1:
        report = response[start_index + len(start_tag):end_index].strip()
    else:
        report = response.strip()
        
    return search_terms, actions, report

async def perform_web_search(query):
    print(colored(f"Performing web search for: {query}", "blue"))
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant performing in-depth web searches. "
                "Provide detailed search results to help answer the research question."
            ),
        },
        {
            "role": "user", 
            "content": query,
        },
    ]

    response = await search_client.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )
    
    return response.choices[0].message.content

async def main():
    research_question = input("Enter the research question: ")
    
    iteration = 1
    while True:
        print(colored(f"\nIteration {iteration}:", "green"))
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"Research Question: {research_question}\n\n"
                    "Instructions:\n"
                    "- Perform web searches to gather information related to the research question.\n"
                    "- Return search terms in the following format: <search_terms>term1, term2, term3</search_terms>\n"  
                    "- Specify actions to take in the following format: <actions>action1, action2, action3</actions>\n"
                    "- If enough information is gathered, write a report in markdown format between ```markdown and ``` tags.\n"
                    "- Continue researching until satisfied with the results.\n"
                )
            }
        ]

        response = await agent_client.chat.completions.create(
            model="o1-mini",
            messages=messages
        )

        agent_response = response.choices[0].message.content
        print(colored(f"Agent Response:\n{agent_response}", "yellow"))
        
        search_terms, actions, report = parse_agent_response(agent_response)
        
        print(colored(f"Search Terms: {search_terms}", "cyan"))
        print(colored(f"Actions: {actions}", "magenta"))
        
        # Perform web searches in parallel
        search_results = await asyncio.gather(*[perform_web_search(term) for term in search_terms])
        
        # Save agent response and search results to JSON files
        with open(f"RTS/iteration_{iteration}_agent.json", "w", encoding="utf-8") as f:
            json.dump({"agent_response": agent_response}, f, ensure_ascii=False, indent=2)
        
        with open(f"RTS/iteration_{iteration}_search_results.json", "w", encoding="utf-8") as f:  
            json.dump({"search_results": search_results}, f, ensure_ascii=False, indent=2)
        
        if report:
            print(colored("Writing research report...", "green"))
            with open("RTS/research_report.md", "w", encoding="utf-8") as f:
                f.write(report)
            break
        
        iteration += 1
        
        print(colored(f"Reasoning Tokens: {response.usage.completion_tokens_details['reasoning_tokens']}", "blue"))

if __name__ == "__main__":
    asyncio.run(main())

