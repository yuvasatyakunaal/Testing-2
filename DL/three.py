# Machine Translation using LSTM (without attention machanism)

import numpy as np
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tiny dataset
eng = ["hello", "goodbye"]
hin = ["<start> नमस्ते <end>", "<start> अलविदा <end>"]

# Tokenize
eng_tok = Tokenizer()
eng_tok.fit_on_texts(eng)

hin_tok = Tokenizer(filters='')
hin_tok.fit_on_texts(hin)

# Prepare data
X = pad_sequences(eng_tok.texts_to_sequences(eng))
y = pad_sequences(hin_tok.texts_to_sequences(hin))

# Model
enc_in = Input(shape=(X.shape[1],))
enc_emb = Embedding(len(eng_tok.word_index)+1, 64)(enc_in)
enc_out, h, c = LSTM(64, return_state=True)(enc_emb)

dec_in = Input(shape=(y.shape[1]-1,))
dec_emb = Embedding(len(hin_tok.word_index)+1, 64)(dec_in)
dec_out, _, _ = LSTM(64, return_sequences=True, return_state=True)(dec_emb, initial_state=[h, c])
out = Dense(len(hin_tok.word_index)+1, activation='softmax')(dec_out)

model = Model([enc_in, dec_in], out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([X, y[:,:-1]], y[:,1:], epochs=300)

# Build encoder model separately
encoder_model = Model(enc_in, [h, c])

# Build decoder model separately
dec_state_input_h = Input(shape=(64,))
dec_state_input_c = Input(shape=(64,))
dec_emb2 = model.layers[3](dec_in)
dec_out2, h2, c2 = model.layers[5](dec_emb2, initial_state=[dec_state_input_h, dec_state_input_c])
dec_out2 = model.layers[6](dec_out2)
decoder_model = Model([dec_in, dec_state_input_h, dec_state_input_c], [dec_out2, h2, c2])

# Translate
def translate(word):
    x = pad_sequences(eng_tok.texts_to_sequences([word]), maxlen=X.shape[1])
    states = encoder_model.predict(x)

    target = np.zeros((1,1))
    target[0] = hin_tok.word_index['<start>']

    result = []
    for _ in range(5):
        output_tokens, h, c = decoder_model.predict([target] + states)
        word_id = np.argmax(output_tokens[0, -1, :])
        word = hin_tok.index_word.get(word_id, '')
        if word == '<end>' or word == '':
            break
        result.append(word)
        target[0] = word_id
        states = [h, c]

    return ' '.join(result)

print(translate("hello"))