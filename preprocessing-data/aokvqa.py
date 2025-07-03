import json
import os
from pathlib import Path

from datasets import load_dataset  # type: ignore
from tqdm import tqdm


def build_question_with_choices(question: str, choices: list[str]) -> str:
    """Combine the question text and multiple-choice options into one string.

    Args:
        question (str): The question text.
        choices (List[str]): A list of four answer options.

    Returns:
        str: The combined question string.
    """
    choice_str = ", ".join([f"{idx} : {choice}" for idx, choice in enumerate(choices)])
    return f"{question} The choices are {choice_str}"


def process_split(split_name: str, images_base_dir: Path, output_dir: Path) -> None:
    """Download a split of A-OKVQA, save images locally, and dump the formatted JSON.

    Args:
        split_name (str): One of "train", "validation", or "test".
        images_base_dir (Path): Directory where images will be stored.
        output_dir (Path): Directory where the JSON files will be written.
    """
    # Use HF token from environment if provided, else fall back to unauthenticated access.
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split_name)

    images_dir = images_base_dir / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    output_records = []

    for example in tqdm(ds, desc=f"Processing {split_name}"):
        image = example["image"]  # PIL.Image.Image
        qid = example["question_id"]
        img_name = f"{qid}.jpg"
        img_path = images_dir / img_name

        # Save the image only once
        if not img_path.exists():
            image.save(img_path)

        # Gather required fields
        question_text: str | None = example.get("question")
        choices = example.get("choices")
        raw_idx = example.get("correct_choice_idx")
        rationales = example.get("rationales", [])
        direct_answers = example.get("direct_answers")

        # Discard examples violating mandatory field presence rules
        if not question_text or image is None or not choices:
            # Essential information missing
            continue

        if split_name != "test":
            # For train/validation insist on full annotation
            if raw_idx is None or not rationales or not direct_answers:
                continue
        else:
            # For test allow null annotations; empty rationales ok as per new rule
            pass

        question_with_choices = build_question_with_choices(question_text, choices)

        answer_idx = int(raw_idx) if raw_idx is not None else None

        output_records.append(
            {
                "image": str(img_path),
                "question": question_with_choices,
                "steps": rationales,
                "answer": str(answer_idx),
            }
        )

    # Write the split to disk
    output_path = output_dir / f"aokvqa_{split_name}.json"
    with open(output_path, "w") as fp:
        json.dump(output_records, fp)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    images_base_dir = data_dir / "images" / "aokvqa"

    # Ensure the base data directory exists
    data_dir.mkdir(exist_ok=True)

    # Ensure the images base directory exists
    images_base_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        process_split(split, images_base_dir, data_dir)


if __name__ == "__main__":
    main() 