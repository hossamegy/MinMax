from deps.DI_vectorstore import bootstrap

deps = bootstrap()
vectorstore = deps['vectorstore']
response = vectorstore.query("What is the capital of France?")
print(response)
