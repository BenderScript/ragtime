import os
import re
import sys
from io import StringIO
from autopep8 import fix_code


# Function to extract code blocks from Markdown
def extract_code_blocks(markdown_text):
    # Updated regular expression
    # This regular expression looks for ```python followed by a newline and #Extract,
    # then captures everything until the closing ```
    code_blocks = re.findall(r'```python\n#Extract\n(.*?)\n```', markdown_text, re.DOTALL)
    return code_blocks


def clean_code(code):
    # Remove comments and empty lines
    return "\n".join(line for line in code.split("\n") if line.strip() and not line.strip().startswith("#"))


# Function to save code block to a file
def save_code_block(code, file_name):
    with open(file_name, 'w') as file:
        file.write(code)


# Function to execute code and capture output
def execute_code(code):
    # Clean the code to remove comments and unnecessary whitespace
    cleaned_code = clean_code(code)
    # Initialize the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Capture the standard output and error
        sys.stdout = captured_output = StringIO()
        sys.stderr = captured_error = StringIO()

        # Execute the code in a new, isolated local scope
        local_scope = {}
        # Execute the cleaned code
        exec(cleaned_code, globals())

        # Get captured output and error
        output_str = captured_output.getvalue()
        error_str = captured_error.getvalue()

        return True, output_str, error_str

    except Exception as e:
        return False, "", str(e)

    finally:
        # Restore the standard output and error
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# Function to process the Markdown file
def process_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    # Parse the Markdown and extract code blocks with language specifier
    code_blocks = extract_code_blocks(markdown_text)

    # Ensure the 'examples' directory exists
    if not os.path.exists('examples'):
        os.makedirs('examples')

    # Execute and report on each code block
    for i, code in enumerate(code_blocks, start=1):
        print(f"Code block {i}:\n")
        print(code)  # Print the code to be executed
        print("\nExecuting...\n")

        # Format the code according to PEP 8
        formatted_code = fix_code(code)

        file_name = f'examples/code_block_{i}.py'
        save_code_block(formatted_code, file_name)

        success, output, error_message = execute_code(code)

        if error_message is not None:
            print(f"Error message:\n{error_message}")
        if success:
            print(f"Output:\n{output}")
            print("Code executed successfully.")
        else:
            print("Code execution failed.")
            print("Error message:")
            print(error_message)

        print("-" * 50)


# Read the Markdown file
markdown_file_path = 'ragtime.md'
process_markdown_file(markdown_file_path)
