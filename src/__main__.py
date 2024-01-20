import os
from pathlib import Path
from llama_index import load_index_from_storage, VectorStoreIndex, StorageContext
from langchain.chat_models.ollama import ChatOllama as ModelLLM
from langchain.schema import HumanMessage, SystemMessage
from langchain.embeddings.ollama import OllamaEmbeddings as ModelEmbedding
from llama_index import SimpleDirectoryReader as SDR
from llama_index import ServiceContext, set_global_service_context
import fire
from .utils import setup_logging

DEBUG = True
PATH_WORKDIR = Path("/home/mcfrank/")
PATH_INPUT = PATH_WORKDIR / "notes"
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
    model=MODEL
)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=llm_embed)
set_global_service_context(service_context)


def process_file(file_path, prompt):
    log(msg=f"running process_file {file_path} with prompt")
    with open(file_path, 'r') as file:
        content = file.read()
        log(msg=f"file content is: {content}")
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"heres the text: {content}")
        ]

        response = llm(messages)
        log(msg=f"final response from llm: {response}")
        return response.content


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
        log(msg=f"working on dir {root}")
        if ".git" in root or "index" in root:
            log(msg=f"skipping dir: {root}")
            continue
        documents = []
        path_dir = Path(root)
        for file in files:
            if not file.endswith(".md"):
                log(msg=f"skipping {file}")
                continue

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

        path_index = path_dir / "index"
        if not os.path.exists(path_index):
            index = VectorStoreIndex.from_documents(
                documents
            )
            index.storage_context.persist(path_index)
        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=path_index
            )
            index = load_index_from_storage(storage_context=storage_context)
            index.docstore.add_documents(documents)
            index.storage_context.persist()


def cli():
    fire.Fire(represent)


if __name__ == "__main__":
    cli()
