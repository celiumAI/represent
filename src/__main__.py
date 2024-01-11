import os
from pathlib import Path
from llama_index import VectorStoreIndex
from llama_index.llms import Ollama as ModelLLM
from llama_index.llms import ChatMessage
from llama_index.embeddings import OllamaEmbedding as ModelEmbedding
from llama_index import SimpleDirectoryReader as SDR
from llama_index import ServiceContext, set_global_service_context
import fire
from .utils import setup_logging

DEBUG = True
PATH_WORKDIR = Path("/home/mcfrank/")
PATH_INPUT = PATH_WORKDIR / "notes/notes"
PATH_PROMPTS = PATH_WORKDIR / "represent/prompts"
DEFAULT_NAME_PROMPT = "keywords"
EXTENSION_TEXT = ".md"
URL_BASE = "http://192.168.2.177:11434"
MODEL = "mistral:instruct"

log = setup_logging(DEBUG)


llm = ModelLLM(
    base_url=URL_BASE,
    model=MODEL
)

llm_embed = ModelEmbedding(
    base_url=URL_BASE,
    model_name=MODEL
)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=llm_embed)
set_global_service_context(service_context)


def process_file(file_path, prompt):
    log(msg=f"running process_file {file_path} with prompt")
    with open(file_path, 'r') as file:
        content = file.read()
        log(msg=f"file content is: {content}")
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=f"heres the text: {content}")
        ]

        response = llm.chat(messages)
        log(msg=f"final response from llm: {response}")
        return response.message.content


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

    log(msg=f"processing all files in {path_input}")
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
