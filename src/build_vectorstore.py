from deps.DI_vectorstore import bootstrap

def run_vectorstore():
    deps = bootstrap()
    
    config = deps["config"]
    loader = deps["loader"]
    chunker = deps["chunker"]
    vectorstore = deps["vectorstore"]

    print(f"Starting application with embedding_type: {config.embedding_type}")

    data = loader.load()
    print(f"Loaded {len(data)} items from {config.data_path}")

    chunks = chunker.chunk(data)
    print(f"Created {len(chunks)} chunks.")

    print("Adding chunks to vector store...")
    vectorstore.add_chunks(chunks)
    print("Done!")

if __name__ == "__main__":
    run_vectorstore()
