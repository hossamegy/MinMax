from src.dep.DI_vectorstore import bootstrap

def main():
    # Initialize dependencies via the DI container
    deps = bootstrap()
    
    config = deps["config"]
    loader = deps["loader"]
    chunker = deps["chunker"]
    vectorstore = deps["vectorstore"]

    print(f"Starting application with embedding_type: {config.embedding_type}")

    # 1. Load Data
    data = loader.load()
    print(f"Loaded {len(data)} items from {config.data_path}")

    # 2. Chunk Data
    chunks = chunker.chunk(data)
    print(f"Created {len(chunks)} chunks.")

    # 3. Store in Vector Database
    print("Adding chunks to vector store...")
    vectorstore.add_chunks(chunks)
    print("Done!")

if __name__ == "__main__":
    main()
