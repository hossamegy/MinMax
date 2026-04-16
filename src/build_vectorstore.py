from dep.DI_vectorstore import bootstrap

container = bootstrap()

vectorstore = container['vectorstore']
vectorstore.add_chunks(container['chunker'].chunk(container['loader'].load(container['config'].data_path)))

