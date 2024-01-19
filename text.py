file_path = 'log.txt'
import re
from pynput.keyboard import Key
try:
    with open(file_path, 'r') as file:
        file_content = file.read()
        print("File content as a string:")
        # Extract words between 'Key.space'
        words = re.findall(r"'([^']+)'|Key\.space", file_content)
        # Combine words, excluding 'Key.space'
        resulting_string = ''
        current_word = ''
        for word in words:
            if word == 'Key.space':
                resulting_string += current_word + '-'
                current_word = ''
            else:
                current_word += word
        # Add the last word
        resulting_string += current_word
        resulting_string = resulting_string.replace('\\x13', '')
        print(resulting_string)
except FileNotFoundError:
    print(f"File not found at path: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")