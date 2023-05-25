from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
print(torch.cuda.is_available())
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import json

def train_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased')
    
    # label mapping
    label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    
    model = BertForTokenClassification.from_pretrained('bert-large-cased', num_labels=9)
    model.config.id2label = label_map
    model.config.label2id = {v: k for k, v in label_map.items()}

    model.config.gradient_checkpointing = True

    datasets = load_dataset("conll2003")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)

        labels_sequence = examples["ner_tags"]

        word_ids = tokenized_inputs.word_ids()
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels_sequence[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        if len(label_ids) != len(tokenized_inputs['input_ids']):
            label_ids = label_ids[:len(tokenized_inputs['input_ids'])]

        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs


    print(datasets["train"][0])
    print(datasets["train"][1])
    print(datasets["train"][2])

    tokenized_datasets = datasets.map(tokenize_and_align_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16, 
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=2, # gradient accumulation
        fp16=True, # mixed precision training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    with open('label2id.json', 'w') as f:
        json.dump(model.config.id2label, f)

    trainer.save_model("fine-tuned-model")

if __name__ == "__main__":
    train_model()
