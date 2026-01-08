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