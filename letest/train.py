from transformers import Trainer, TrainingArguments
import torch

def setup_trainer(model, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    data_collator = lambda data: {
        'x': torch.stack([d['x'] for d in data]),
        'labels': torch.stack([d['labels'] for d in data])
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},
        data_collator=data_collator
    )

    return trainer
