import json

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

def extract_json_from_llm_response(response: str) -> dict:
    """
    Extracts and returns the first JSON object found in the given LLM response text.
    
    :param response_text: The text response from the LLM which may contain a JSON object.
    :return: The extracted JSON object as a dictionary.
    :rtype: dict
    :raises ValueError: If no valid JSON object is found in the response text.
    """
    try:            
        cleaned_response = response.strip()
        if cleaned_response.startswith('.'):
            cleaned_response = cleaned_response[1:].strip()

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