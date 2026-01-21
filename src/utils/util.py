import json
from openai import OpenAI

# This method converts a TinyDB database into a llm readable string format
def dbToString(db):
        allEntries = db.all()
        
        sections_dict = {}
        for entry in allEntries:
            section = entry['section']
            if section not in sections_dict:
                sections_dict[section] = []
            sections_dict[section].append(entry)
        
        output = []
        for section, entries in sections_dict.items():
            output.append(f"## {section}")
            output.append("")
            for entry in entries:
                output.append(f"- {entry['id']}: {entry['content']}")
                output.append("")
        
        return "\n".join(output)

# check msg.content and msg.tool_calls when calling this method    
def call_llm(client: OpenAI, model: str, prompt: list[dict], tool_calls: list[dict] = None, debug=False): 
    try:
        response = client.chat.completions.create(model=model, messages=prompt, tools=tool_calls)
        if not response.choices:
            raise ValueError("No choices returned from LLM")
        
        msg = response.choices[0].message
        if not msg:
            raise ValueError("No message returned from LLM choice")
        
        if debug:
            print(f"Agent Response: {msg}")
            
        return msg
            
    except Exception as e:
        print(f"Unexpected error when parsing llm response: {type(e).__name__}: {e}")
        return None

def extract_json_from_llm_response(response: str) -> dict:
    """
    Extracts and returns the first JSON object found in the given LLM response text.
    
    :param response_text: The text response from the LLM which may contain a JSON object.
    :return: The extracted JSON object as a dictionary.
    :rtype: dict
    :raises ValueError: If no valid JSON object is found in the response text.
    """
               
    cleaned_response = response.strip()
    if cleaned_response.startswith('.'):
        cleaned_response = cleaned_response[1:].strip()
    try:
        response_dict = json.loads(cleaned_response)
        return response_dict
    except json.JSONDecodeError:
        try:
            start_idx = cleaned_response.find('{')
            end_idx = cleaned_response.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_str = cleaned_response[start_idx:end_idx+1]
                response_dict = json.loads(json_str)
                return response_dict
            else:
                raise json.JSONDecodeError("No JSON object found", cleaned_response, 0)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return response
        
def execute_tool_call(TOOL_MAPPING, tool_call, debug=False):
    tool_name = tool_call.function.name
    tool_response = None
    
    # Validate tool exists
    if tool_name not in TOOL_MAPPING:
        if debug:
            print(f"== Tool '{tool_name}' not found in TOOL_MAPPING. Skipping.")
        return tool_response
    
    arguments = tool_call.function.arguments
    if arguments is None or not arguments.strip():
        try:
            tool_response = TOOL_MAPPING[tool_name]()
        except Exception as e:
            if debug:
                print(f"== Error executing tool '{tool_name}' with no arguments: {e}")
    else:
        try:
            parsed_args = extract_json_from_llm_response(arguments)
            if isinstance(parsed_args, dict):
                tool_response = TOOL_MAPPING[tool_name](**parsed_args)
            else:
                # Parsing failed (extract_json_from_llm_response returned raw string)
                if debug:
                    print(f"== Failed to parse arguments for tool '{tool_name}': {arguments}")
        except Exception as e:
            if debug:
                print(f"== Error executing tool '{tool_name}': {e}")
                print(f"== Raw arguments: {arguments}")
    
    if debug:
        print(f"== Tool: {tool_name}")
        print(f"== Tool Parameters: {arguments}")
    
    return tool_response