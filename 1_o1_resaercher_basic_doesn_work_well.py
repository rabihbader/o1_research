import asyncio
import json
import os
from termcolor import colored
from openai import AsyncOpenAI
import re

# Initialize OpenAI clients
openai_client = AsyncOpenAI()
perplexity_client = AsyncOpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")

# Create the research folder if it doesn't exist
os.makedirs("research", exist_ok=True)

async def get_agent_response(messages):
    print(colored("Getting agent response...", "cyan"))
    response = await openai_client.chat.completions.create(
        model="o1-mini",
        messages=messages
    )
    content = response.choices[0].message.content
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    print(colored(f"Reasoning tokens: {reasoning_tokens}", "yellow"))
    return content

async def perform_web_search(query):
    print(colored(f"Performing web search: {query}", "magenta"))
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant tasked with performing in-depth web searches and returning detailed results."
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
    search_terms = re.findall(r'<search>(.*?)</search>', response)
    actions = re.findall(r'<action>(.*?)</action>', response)
    return search_terms, actions

async def process_agent_response(response, messages, iteration):
    search_terms, actions = parse_agent_response(response)
    
    if search_terms:
        print(colored(f"Search terms: {search_terms}", "green"))
        search_results = await asyncio.gather(*[perform_web_search(term) for term in search_terms])
        
        # Save search results
        with open(f"research/iteration_{iteration}_search_results.json", "w", encoding="utf-8") as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)
        
        # Append search results to messages
        for term, result in zip(search_terms, search_results):
            messages.append({"role": "user", "content": f"Search results for '{term}': {result}"})
    
    for action in actions:
        if action.lower().startswith("write report"):
            write_report(response)
            return True  # End the research loop
    
    return False  # Continue the research loop

def write_report(response):
    print(colored("Writing research report...", "green"))
    report_content = re.search(r'```markdown(.*?)```', response, re.DOTALL)
    if report_content:
        report_content = report_content.group(1).strip()
    else:
        report_content = response
    
    with open("research/research_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

async def main():
    print(colored("Starting the research AI agent...", "blue"))
    research_question = input(colored("Enter your research question: ", "yellow"))
    
    messages = [
        {
            "role": "user",
            "content": f"""You are an advanced AI research agent. Your task is to investigate the following question: '{research_question}'

            Follow these guidelines:
            1. Conduct thorough research by suggesting search terms enclosed in <search></search> tags.
            2. Analyze the search results and determine the next steps.
            3. When you have gathered enough information, write a comprehensive report using the <action>write report</action> tag. Only return this after you have seen adequte search results to be abelt o effectivelyd answer the question
            4. Format your report in markdown, enclosed in ```markdown and ``` tags.
            5. Be autonomous and continue researching until you are satisfied with the results.

            Begin your research process now."""
        }
    ]
    
    iteration = 1
    while True:
        response = await get_agent_response(messages)
        
        # Save agent response
        with open(f"research/iteration_{iteration}_agent.json", "w", encoding="utf-8") as f:
            json.dump({"response": response}, f, ensure_ascii=False, indent=2)
        
        messages.append({"role": "assistant", "content": response})
        
        if await process_agent_response(response, messages, iteration):
            break
        
        iteration += 1

if __name__ == "__main__":
    asyncio.run(main())

