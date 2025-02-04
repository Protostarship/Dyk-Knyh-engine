import json

def parse_dictionary(content):
    # Split content into lines and remove empty lines
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    # Initialize dictionary
    dictionary = {}
    
    # Parse each line
    for line in lines:
        # Split by tab
        parts = line.split('\t')
        if len(parts) == 3:
            # Get the index, indigenous word, and Indonesian word
            index = parts[0]
            indigenous = parts[2]  # Indigenous language is in column 3
            indonesian = parts[1]  # Indonesian language is in column 2
            
            # Add to dictionary using index as key
            dictionary[index] = {
                "indigenous": indigenous,
                "indonesian": indonesian
            }
    
    return dictionary

# Example content string (you would replace this with actual file reading)
content = """
0	satu	ca
1	dua	dua
2	tiga	telu
3	empat	pat
4	lima	lema
"""  # This is just an example, you'd use the full content

def main():
    # In practice, you'd read from a file like this:
    with open('distil\\table.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parse the content
    dictionary = parse_dictionary(content)
    
    # Write to JSON file
    with open('indigenous_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
    
    # Print first few entries as example
    print("First few entries of the parsed dictionary:")
    first_entries = dict(list(dictionary.items())[:5])
    print(json.dumps(first_entries, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()