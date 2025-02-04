import json

def reformat_dictionary(indigenous_dict):
    """
    Reformats the indigenous dictionary to match the target format
    """
    reformatted = {}
    
    # Process each entry in the indigenous dictionary
    for entry in indigenous_dict.values():
        indonesian_word = entry['indonesian']
        indigenous_word = entry['indigenous']
        
        # Create the new format
        reformatted[indonesian_word] = {
            "indigenous": indigenous_word,
            "pos": "noun"  # Default to noun since original doesn't specify pos
        }
    
    return reformatted

# Sample usage
if __name__ == "__main__":
    # Load the indigenous dictionary
    with open('indigenous_dictionary.json', 'r') as f:
        indigenous_dict = json.load(f)
    
    # Load the existing dictionary to append to
    with open('dictionary.json', 'r') as f:
        existing_dict = json.load(f)
    
    # Reformat the indigenous dictionary
    reformatted_dict = reformat_dictionary(indigenous_dict)
    
    # Merge with existing dictionary
    existing_dict.update(reformatted_dict)
    
    # Save the combined dictionary
    with open('dictionary_combined.json', 'w', encoding='utf-8') as f:
        json.dump(existing_dict, f, ensure_ascii=False, indent=2)