#
# Generic MCP client, communicating with an MCP server running sse,
# and using a textbased LLM interface
#

import asyncio
from fastmcp import Client, exceptions
from fastmcp.client.transports import SSETransport
import base64
import json

import openai
from openai import OpenAI

ollama_config = {
    "model": "llama3.1",
    "base_url": "http://localhost:11434/v1/",
    "api_key": "ollama"
}

messages_trunclen = 4
messages = []

has_sysprompt = False
has_sysprompt_lang = False
has_augprompt = False
has_augprompt_lang = False
has_name = False
has_init = False
has_exit = False

def init_llm(conf):
    default_config = conf
    if "api_key" in default_config:
        openai.api_key = default_config["api_key"]
    if "base_url" in default_config:
        openai.base_url = default_config["base_url"]
    model = default_config["model"]
    llm = OpenAI(api_key=openai.api_key)
    return (llm, model)

def map_tool_definition(f):
        tool_param = {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.inputSchema,
            },
        }
        return tool_param

async def system_message(client, lang):
    if has_sysprompt:
        if has_sysprompt_lang:
            pr = await client.get_prompt("get_service_prompt", {"lang": lang})
        else:
            pr = await client.get_prompt("get_service_prompt", {})
        txt = pr.messages[0].content.text
    else:
        txt = "You are a helpful assistant that can control various devices."
    return {"role": "system", "content": txt}

async def augmentation_message(client, lang):
    if has_augprompt:
        if has_augprompt_lang:
            pr = await client.get_prompt("get_service_augmentation", {"lang": lang})
        else:
            pr = await client.get_prompt("get_service_augmentation", {})
        txt = pr.messages[0].content.text
        return {"role": "system", "content": txt}
    else:
        return False

def user_message(prompt):
    return {"role": "user", "content": prompt}

def compose_messages(sysp, mlst, aug, lang):
    n = 0
    i1 = 0
    i2 = 0
    for i in reversed(range(len(mlst))):
        if type(mlst[i])==dict and mlst[i]["role"] == 'user':
            n += 1
            if n == 1:
                i2 = i
            if n == messages_trunclen:
                i1 = i
                break
    return [sysp] + mlst[i1:i2] + ([aug] if aug else []) + mlst[i2:]

def clear_messages():
    global messages
    messages = []

def trim_last_message():
    global messages
    for i in reversed(range(len(messages))):
        if type(messages[i])==dict and messages[i]["role"] == 'user':
            messages = messages[0:i+1]
            return True
    return False


async def main():
    global messages
    global tools
    global has_sysprompt
    global has_sysprompt_lang
    global has_augprompt
    global has_augprompt_lang
    global has_name
    global has_init
    global has_exit

    # Connect via stdio to a local script
    async with Client(transport=SSETransport("http://127.0.0.1:8000/sse")) as client:
        # Check MCP server capabilities
        ress = await client.list_resources()
        print("\nAvailable resources:")
        for res in ress:
            print(res)
            if res.name == 'url://service_name':
                has_name = True
            if res.name == 'url://service_init':
                has_init = True
            if res.name == 'url://service_exit':
                has_exit = True
        
        prompts = await client.list_prompts()
        print("\nAvailable prompts:")
        for prompt in prompts:
            print(prompt)
            if prompt.name == 'get_service_prompt':
                has_sysprompt = True
                for arg in prompt.arguments:
                    if arg.name == 'lang':
                        has_sysprompt_lang = True
            if prompt.name == 'get_service_augmentation':
                has_augprompt = True
                for arg in prompt.arguments:
                    if arg.name == 'lang':
                        has_augprompt_lang = True

        tools = await client.list_tools()
        print("\nAvailable tools:")
        for tool in tools:
            print(tool)
        tools = [map_tool_definition(tool) for tool in tools]
        print("\n")

        # Initialization

        llm, model = init_llm(ollama_config)
        print(f'LLM Chatbot using model {model}')

        if has_name:
            tmp = await client.read_resource("url://service_name")
            name = tmp[0].text
        else:
            name = "MCP Speech Client"

        if has_init:
            ok = await client.read_resource("url://service_init")
            if ok:
                if has_name:
                    print('Initialized service '+name)
                else:
                    print('Initialized service')
            else:
                print('Failed to initialize service')
                return False

        # Main loop 

        repeat = False
        lang = False
        prompt = ""
        txtlang = 'en'
        sysprompt = False
        augprompt = False
        while True:
            print("")
            res = input("User: ")
            if res:
                res = res.strip(" \n")
                if res[0:5] == "/lang":
                    txtlang = res[5:].strip(" ")
                    continue
                elif res[0:5] == "/exit":
                    break
                elif res[0:6] == "/clear":
                    clear_messages()
                    print("\n  (Cleared history)")
                    continue
                elif res[0:7] == "/repeat":
                    if sysprompt and prompt and lang and trim_last_message():
                        print("\n  (Repeating)")
                        repeat = True
                    else:
                        continue
                elif len(res):
                    prompt = res
                    lang = txtlang
                else:
                    continue

                if not repeat:
                    sysprompt = await system_message(client, lang)
                    augprompt = await augmentation_message(client, lang)
                    if augprompt:
                        print("\n  Augmentation:")
                        print(augprompt['content'])
                    # print("\n  User: (", lang, ") ", prompt)
                    messages.append(user_message(prompt))
                else:
                    repeat = False

                response = openai.chat.completions.create(
                    model=model,
                    messages=compose_messages(sysprompt, messages, augprompt, lang),
                    tools=tools,
                )
    
                tool_calls = response.choices[0].message.tool_calls
                while tool_calls:
                    messages.append(response.choices[0].message)
                    for tool_call in tool_calls:
                        try:
                            result = await client.call_tool(tool_call.function.name,
                                                            json.loads(tool_call.function.arguments))
                            result_message = {
                                "role": "tool",
                                "content": json.dumps({
                                    "result": result[0].text
                                }),
                                "tool_call_id": tool_call.id
                            }
                            print("\n  Function: ", tool_call.function.name, "(", tool_call.function.arguments, ")")
                            print(  "  Result:   ", result[0].text)
                            messages.append(result_message)
                        except exceptions.ToolError:
                            result_message = {
                                "role": "tool",
                                "content": json.dumps({
                                    "result": "unknown function called"
                                }),
                                "tool_call_id": tool_call.id
                            }
                            print("\n  Unknown function: ", tool_call.function.name, "(", tool_call.function.arguments, ")")
                            messages.append(result_message)
    
                    response = openai.chat.completions.create(
                        model=model,
                        messages=compose_messages(sysprompt, messages, augprompt, lang),
                        tools=tools,
                    )
                    tool_calls = response.choices[0].message.tool_calls
    
                # No tool calls, just print the response.
                messages.append(response.choices[0].message)
                print(f'\n  Response: {response.choices[0].message.content}')
    
        if has_exit:
            ok = await client.read_resource("url://service_exit")
        print('Exiting')

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
