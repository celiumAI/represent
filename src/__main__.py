import os
from pathlib import Path
from langchain.llms import Ollama as LLM
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from llama_index import VectorStoreIndex
from llama_index import SimpleDirectoryReader as SDR
import fire
from .utils import setup_logging

DEBUG = True
PATH_WORKDIR = Path("/home/mcfrank/")
PATH_INPUT = PATH_WORKDIR / "notes/notes"
PATH_PROMPTS = PATH_WORKDIR / "represent/prompts"
DEFAULT_NAME_PROMPT = "keywords"
EXTENSION_TEXT = ".md"

log = setup_logging(DEBUG)

llm = LLM(
    base_url="http://192.168.2.177:11434",
    model="mistral:instruct",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)


def process_file(file_path, prompt):
    log(msg=f"running process_file {file_path} with prompt")
    with open(file_path, 'r') as file:
        content = file.read()
        log(msg=f"file content is: {content}")
        response = llm(f"{prompt}\n\n{content}")
        log(msg=f"final response from llm: {response}")
        return response


def represent(path_input: str = PATH_INPUT,
              name_prompt: str = DEFAULT_NAME_PROMPT):

    log(msg="reading prompt file")
    path_prompt = PATH_PROMPTS / (name_prompt + EXTENSION_TEXT)
    with open(path_prompt, 'r') as prompt_file:
        prompt = prompt_file.read()

    log(msg=f"read in prompt: {prompt}")

    path_output_dir = Path(path_input).parent / name_prompt
    log(msg=f"preparing output directory at {path_output_dir}")
    path_output_dir.mkdir(exist_ok=True)

    # Process each file
    for root, dirs, files in os.walk(path_input):
        log(msg=f"working on {root}")
        documents = []
        for file in files:
            if file.endswith(".embeddings"):
                log(msg=f"skipping {file}")
                continue

            path_dir = Path(root)
            file_path = path_dir / file
            relative_path = file_path.relative_to(path_input)
            output_file_path = path_output_dir / relative_path

            if output_file_path.exists():
                log(msg=f"already exists! skipping {output_file_path}")
                continue

            log(msg=f"working on {file_path}")
            result = process_file(file_path, prompt)
            temp_docs = SDR(root).load_data(show_progress=True)
            log(msg=f"loaded docs {temp_docs}")
            documents += temp_docs
            # Save the result in a mirrored structure
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, 'w') as output_file:
                output_file.write(result)

        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(path_dir)


if __name__ == "__main__":
    fire.Fire(represent)
