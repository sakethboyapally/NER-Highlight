from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def get_entities(text):
    # model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    model_name = "fine-tuned-model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    results = nlp(text)
    print("Raw results:", results)

    entities = {'person': [], 'location': [], 'organization': []}
    current_entity = {'type': None, 'tokens': []}

    for result in results:
        entity_type = result['entity'].split('-')[-1].lower()
        word = result['word']

        # Handle subword tokens
        if word.startswith("##"):
            if not current_entity['tokens']:
                current_entity['tokens'].append(word[2:])
            else:
                current_entity['tokens'][-1] += word[2:]
        else:
            if current_entity['tokens'] and (current_entity['type'] is not None):
                entities[current_entity['type']].append(' '.join(current_entity['tokens']))
            current_entity = {'type': None, 'tokens': [word]}

        if entity_type == 'per':
            current_entity['type'] = 'person'
        elif entity_type == 'loc':
            current_entity['type'] = 'location'
        elif entity_type == 'org':
            current_entity['type'] = 'organization'

    if current_entity['tokens'] and (current_entity['type'] is not None):
        entities[current_entity['type']].append(' '.join(current_entity['tokens']))

    return entities

# text = ""
# entities = get_entities(text)
# print(entities)