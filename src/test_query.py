from deps.DI_vectorstore import bootstrap

deps = bootstrap()
vectorstore = deps['vectorstore'].get_vectorstore()
print(vectorstore.similarity_search("What is the capital of France?"))