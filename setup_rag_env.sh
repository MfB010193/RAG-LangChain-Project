#!/usr/bin/env bash
set -e

ENV_NAME="rag_env"
PY_VERSION="3.10"

echo "Creating conda environment: $ENV_NAME (python $PY_VERSION)"
conda create -n $ENV_NAME python=$PY_VERSION -y

echo "Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing packages for RAG..."
pip install "langchain==0.2.11" \
            "langchain-community==0.2.11" \
            "langchain-openai==0.1.6" \
            "openai==1.109.1" \
            "tiktoken==0.12.0" \
            "faiss-cpu==1.13.1" \
            "pypdf==3.11.0" \
            "pandas==2.1.3" \
            "ipykernel" \
            "notebook"

echo "Registering kernel for Jupyter"
python -m ipykernel install --user --name $ENV_NAME --display-name "RAG Environment (rag_env)"

echo "Setup complete!"
