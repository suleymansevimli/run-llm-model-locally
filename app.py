from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

model = AutoModelForTokenClassification.from_pretrained('PATH_OF_YOUR_LOCAL_MODEL')
tokenizer = AutoTokenizer.from_pretrained('PATH_OF_YOUR_LOCAL_MODEL')

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Example Text."
ner_results = nlp(example)
print(ner_results)