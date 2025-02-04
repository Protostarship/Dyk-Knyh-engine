import json

def create_json_dictionary(file_path):
    # Read the text file contents
    with open(file_path, 'r') as f:
        text = f.read()
    
    # Split the text into words using whitespace as a delimiter
    words = text.split()
    
    # Create a JSON list with numbering for each word
    json_list = []
    for index, word in enumerate(words, start=1):
        json_list.append({"id": index, "word": word})
    
    return json_list

if __name__ == '__main__':
    input_file = 'distil\\indonesian_words.txt'   # Replace with your text file path
    output_file = 'INDO_reformatted.json'  # Output JSON file

    # Create the JSON dictionary list from the file
    dictionary = create_json_dictionary(input_file)
    
    # Write the resulting JSON list to a file with indentation for readability
    with open(output_file, 'w') as f:
        json.dump(dictionary, f, indent=4)
    
    print(f"JSON dictionary has been created and saved to {output_file}")
