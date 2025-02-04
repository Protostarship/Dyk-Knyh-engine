import csv

# Set these paths as needed
input_csv = 'E:\\DYK_KNYH\\DYK_Machine\\distil\\pbwl_dataset.csv'     # the CSV file you exported
output_txt = 'E:\\DYK_KNYH\\DYK_Machine\\distil\\indonesian_words.txt'  # the output file

# Using "Root" as the column name
word_column = 'Root'

words = set()  # using a set to avoid duplicates

try:
    with open(input_csv, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Check if the required column exists
        if word_column not in reader.fieldnames:
            raise KeyError(f"Column '{word_column}' not found in CSV. Available columns: {', '.join(reader.fieldnames)}")
        
        for row in reader:
            # Get the word from the Root column
            word = row[word_column].strip()
            
            # Skip empty entries
            if not word:
                continue
                
            # Split on " `-`" and take the first part
            if " `-`" in word:
                word = word.split(" `-`")[0].strip()
            elif " -" in word:  # Fallback for simple hyphen
                word = word.split(" -")[0].strip()
                
            # Add non-empty words to the set
            if word:
                words.add(word)

    # Sort words alphabetically
    sorted_words = sorted(words)

    # Join all words with a single whitespace
    output_line = " ".join(sorted_words)

    # Write to output file
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(output_line)

    print(f"Successfully extracted {len(sorted_words)} unique words into {output_txt}")
    
except FileNotFoundError:
    print(f"Error: Could not find the input file '{input_csv}'")
except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")