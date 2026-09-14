from tokenizer.tokenizer import BasicTokenizer


tokenizer = BasicTokenizer()

english = '"What does Bessie say I have done?" I asked.'

german = '»Was sagt denn Bessie, daß ich gethan habe?« fragte ich.'

print("ENGLISH:")
print(tokenizer.tokenize(english))

print()

print("GERMAN:")
print(tokenizer.tokenize(german))