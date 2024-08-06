import os
import fire
from .broker.llm import llm
from .broker.models import *
from .broker.app import *
from .broker.ingest import ingest_repo_notes
from .utils import setup_logging

DEBUG = True
PATH_INPUT = os.environ.get("PATH_NOTES", "notes")
PATH_PROMPTS = os.environ.get("PATH_PROMPTS", "prompts")
DEFAULT_NAME_PROMPT = "keywords"
EXTENSION_TEXT = ".md"
URL_LLM = os.environ.get("URL_LLM", "http://localhost:11434")
MODEL = "mistral:instruct"

log = setup_logging(DEBUG)


def represent_text(content, prompt):
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": "heres the text to rethink and rephrase:" + content
        }
    ]

    response = llm(messages)
    log(msg=f"final response from llm: {response}")
    return response


async def represent(path_input: str = PATH_INPUT,
                    name_prompt: str = DEFAULT_NAME_PROMPT):

    log(msg="reading prompt file")

    # path_prompt = PATH_PROMPTS / (name_prompt + EXTENSION_TEXT)

    path_prompt = os.path.join(PATH_PROMPTS, name_prompt + EXTENSION_TEXT)
    with open(path_prompt, 'r') as prompt_file:
        prompt = prompt_file.read()

    log(msg=f"read in prompt: {prompt}")

    path_output_dir = os.path.join(os.path.dirname(path_input), name_prompt)

    log(msg=f"preparing output directory at {path_output_dir}")

    os.makedirs(path_output_dir, exist_ok=True)

    log(msg=f"processing all notes in {path_input}")

    notes = ingest_repo_notes(path_input)

    for note in notes:
        represented = represent_text(note.content, prompt)

        represented_note = Note(
            type="note:represented",
            h0=f"{name_prompt} {note.h0}",
            timestamp=note.timestamp,
            origin="represent",
            author=MODEL,
            content=represented
        )

        await make_note(represented_note)
        await make_link(
            Link(
                source=note.node_id,
                target=represented_note.node_id
            )
        )


def cli():
    # if -n is present run the ingest function first
    fire.Fire(represent)


if __name__ == "__main__":
    cli()
