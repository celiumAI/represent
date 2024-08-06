# Use a slim Python image as the base image
FROM python:3.12-slim as builder

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /represent

# Copy all the files from the current directory to the working directory in the container
COPY notes ./notes
COPY prompts ./prompts
COPY src ./src
COPY setup.py .
COPY README.md .

# Install the package in editable mode
RUN pip install -e .

# Set default values for environment variables (optional)
# this cant worktheres no llm in here
ENV HOST_LLM=localhost
ENV PORT_LLM=11434
ENV HOST_BROKER=localhost
ENV PORT_BROKER=5050
ENV HOST=0.0.0.0
ENV PORT=5000

# Run the represent command
CMD ["represent"]
