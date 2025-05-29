# Prosus AI

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Demo

To run the demo application, use the following command:

```bash
streamlit run demo.py
```

## Features

- **`process_data`**: Extracts metadata and full text from the input data.  
- **`embeddings_openai`**: Generates embeddings for the full text.  
- **`embedding_name`**: Generates embeddings specifically for the product name.
- **`evaluation`**: Evaluates rerank vs without rerank (only cosine sim). Also able to evaluate different embeddings by changing embedding file
