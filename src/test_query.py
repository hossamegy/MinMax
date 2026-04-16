from dep.DI_vectorstore import bootstrap

container = bootstrap()
vectorstore = container['vectorstore']
response = vectorstore.query("What is the capital of France?")
print(response)
