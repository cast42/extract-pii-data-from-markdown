import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import torch
from gliner import GLiNER  # type: ignore  # noqa: PGH003
from gliner.model import BaseModel, BaseORTModel  # type: ignore  # noqa: PGH003
from rich.progress import Progress

MAX_LENGHT = 2048

# Set the logging level
logging.basicConfig(level=logging.INFO)


def extract_pii_data(  # type: ignore  # noqa: PGH003
    sentence: str, model: Optional[Union[BaseModel, BaseORTModel]]
) -> list[dict]:
    """
    Extract Personally Identifiable Information (PII) data from a sentence.

    Args:
        sentence (str): The sentence to analyze for PII.
        model: The GLiNER model used for entity prediction.

    Returns:
        List[Dict[str, Union[str, bool]]]: A list of dictionaries, each containing
                                           detected PII type, value, and privacy status.
    """

    pii_data: list[dict] = []

    if model is None:
        # Handle the case where no model is provided
        return pii_data
    labels = [
        "name",
        "last_name",
        "first_name",
        "email",
        "location",
        "url",
        "street_address",
        "company_name",
        "function title",
        "account_number",
        "phone_number",
    ]
    entities = model.predict_entities(sentence, labels, threshold=0.4)
    for entity in entities:
        print(f'{entity["text"]} => {entity["label"]}, {entity["score"]}')
        if entity["score"] > 0.5:
            pii_data.append({
                "pii_type": entity["label"],
                "pii_value": entity["text"],
                "private": True,
            })

    return pii_data


def split_text(text: str, max_length: int = MAX_LENGHT) -> list[str]:
    """
    Split a given text into chunks of up to `max_length` characters, ensuring
    that words are not split mid-way. Each chunk will end at a space if possible,
    and will contain complete words only.

    Args:
        text (str): The text to be split into chunks.
        max_length (int): The maximum length of each chunk. Defaults to 512.

    Returns:
        list[str]: A list of text chunks, each up to `max_length` characters long.
    """
    words = text.split()
    chunks: list[str] = []
    current_chunk: list[str] = []

    for word in words:
        # Check if adding the next word would exceed the maximum length
        if len(" ".join([*current_chunk, word])) > max_length:
            # If it would, add the current chunk to chunks and start a new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            # Otherwise, add the word to the current chunk
            current_chunk.append(word)

    # Add any remaining words as the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def extract_pii_from_markdown(file_path: Union[str, Path]) -> bool:
    """
    Extract PII data from a Markdown file and save it in JSON Lines format.

    Args:
        file_path (Union[str, Path]): Path to the Markdown file.

    Returns:
        bool: True if extraction succeeded, False otherwise.
    """

    markdown_path = Path(file_path)

    # Check if the file exists
    if not markdown_path.is_file():
        logging.error(f"Error: The file '{file_path}' does not exist.")
        return False

    if markdown_path.suffix != ".md":
        logging.error(f"Error: The file '{file_path}' should be a markdown file with extention '.md'.")
        return False

    jsonl_file_path = markdown_path.with_suffix(".jsonl")

    # Check if the to be generated jsonl file alreadt exists
    if jsonl_file_path.is_file():
        logging.error(f"Error: The file '{jsonl_file_path}' already exist.")
        return False

    # Load the fine-tuned GLiNER model
    # model = GLiNER.from_pretrained("gretelai/gretel-gliner-bi-small-v1.0")
    # model = GLiNER.from_pretrained("gretelai/gretel-gliner-bi-base-v1.0")
    model = GLiNER.from_pretrained("gretelai/gretel-gliner-bi-large-v1.0", max_length=MAX_LENGHT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    document = markdown_path.read_text()
    sentences = document.split("\n\n")
    number_of_sentences = len(sentences)
    all_pii_data: list[dict] = []

    # Using a progress bar with rich
    with Progress() as progress:
        task = progress.add_task("Extracting PII data...", total=number_of_sentences)

        # Iterate over each section of the document
        for i, sentence in enumerate(sentences):
            progress.update(task, description=f"Processing sentence {i}/{number_of_sentences}")
            chunks = split_text(sentence)
            for chunk in chunks:
                pii_data = extract_pii_data(chunk, model)
                if not pii_data:
                    continue
                for new_entry in pii_data:
                    if not any(entry["pii_value"] == new_entry["pii_value"] for entry in all_pii_data):
                        all_pii_data.append(new_entry)

            # Advance the progress bar
            progress.advance(task)

    sorted_all_pii_data = sorted(all_pii_data, key=lambda x: len(x["pii_value"]), reverse=True)
    with open(jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
        for pii_item in sorted_all_pii_data:
            json_line_text = json.dumps(pii_item, ensure_ascii=False)
            jsonl_file.write(json_line_text + "\n")

    return True


def main() -> None:
    """
    Main function for parsing arguments and extracting PII from Markdown files.
    """

    parser = argparse.ArgumentParser(
        description="Check if a Markdown file exists and create list of PII data in JSON lines format."
    )
    parser.add_argument("file_path", type=str, help="Path to the Markdown file")
    args = parser.parse_args()

    if not extract_pii_from_markdown(args.file_path):
        sys.exit(1)  # Exit with an error status if the markdown text can not be read or something went wrong


if __name__ == "__main__":  # pragma: no cover
    main()
