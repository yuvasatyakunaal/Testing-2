import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer

# Dataset
data = {
    'text': [
        'Excellent product!', 'Poor quality', 'Highly recommend',
        'Mediocre, not bad', 'Worst purchase ever', 'It works fine'
    ],
    'label': [1, 0, 1, 0, 0, 1]
}

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(data['text'], truncation=True, padding=True, max_length=128, return_tensors='tf')

# Model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.config.id2label = {0: "Negative", 1: "Positive"}

# Create optimizer (huggingface way)
steps_per_epoch = 1  # small data here
num_train_steps = steps_per_epoch * 3  # epochs
optimizer, _ = create_optimizer(
    init_lr=2e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=0
)

# Compile Model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=1)
# Remove reduce_lr callback completely
history = model.fit(
    {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
    tf.convert_to_tensor(data['label']),
    epochs=3
)


# Save Model
model.save_pretrained("my_bert_model")
tokenizer.save_pretrained("my_bert_model")

# Inference
def predict(text):
    inputs = tokenizer(text, return_tensors='tf')
    logits = model(inputs).logits
    probabilities = tf.nn.softmax(logits).numpy()[0]
    return {
        "label": model.config.id2label[int(tf.argmax(logits, axis=1))],
        "confidence": float(max(probabilities))
    }

# Example
print(predict("Very good"))
