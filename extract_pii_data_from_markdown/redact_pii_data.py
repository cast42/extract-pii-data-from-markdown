import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Union

logging.basicConfig(level=logging.INFO)


def remove_pii_from_markdown(markdown_file_path: Union[str, Path], pii_data_file_path: Union[str, Path]) -> bool:
    markdown_file = Path(markdown_file_path)
    pii_data_file = Path(pii_data_file_path)

    # Check if the markdown file exists and has the correct extension
    if not markdown_file.is_file() or markdown_file.suffix != ".md":
        print(f"Error: The file '{markdown_file}' does not exist or is not a markdown (.md) file.")
        return False

    markdown_text = markdown_file.read_text()

    # Iterating over each line in the JSONLines file
    with open(pii_data_file, encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            try:
                # Parse each line as JSON
                data = json.loads(line)

                if data["private"]:
                    # Create a replacement string with black circles (⚫) of the same length as the string to replace
                    replacement = "⚫" * len(data["pii_value"])
                    text_to_replace = re.escape(data["pii_value"])
                    # if text_to_replace.startswith("+"):
                    #     text_to_replace = text_to_replace[1:]

                    # Use re.sub() to replace the matched string with the black circles
                    markdown_text = re.sub(
                        text_to_replace,
                        replacement,
                        markdown_text,
                        flags=re.IGNORECASE,
                    )
            except json.JSONDecodeError:
                # Handle JSON parsing errors
                logging.exception(f"Error parsing line {line_number}")
            except Exception:
                # Handle any other type of error
                logging.exception(f"Error parsing:\n{line}\nResult:\n{data}\nError:\n")

    markdown_file.with_suffix(".redacted.md").write_text(markdown_text)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check if a Markdown file exists and replace PII data present in JSON lines format file."
    )

    parser.add_argument("markdown_file_path", type=str, help="Path to the Markdown file")
    parser.add_argument("jsonl_file_path", type=str, help="Path to the jsonl file")
    args = parser.parse_args()

    if not remove_pii_from_markdown(args.markdown_file_path, args.jsonl_file_path):
        sys.exit(
            1
        )  # Exit with an error status if the markdown and/or the jsonl file is not valid or something went wrong


if __name__ == "__main__":
    main()
