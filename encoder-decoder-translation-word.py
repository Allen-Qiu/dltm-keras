"""
words级的 english-chinese翻译模型。
英文是词级，中文还是字符级。都用embedding。
decoder rnn输出后的处理按照下面论文的相关部分来实施
Learning Phrase Representations using RNN Encoder-Decoder
for Statistical Machine Translation
@author: Allen Qiu
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential

import numpy as np

lr = 0.001        # learning rate
batch_size = 50   # Batch size for training.
epochs = 90       # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
latent_dim2 = 128
vocab_size = 5000 # english vocabulary
embed_size = 128
data_path = '../data/cmn.txt'

# 1. build dataset
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[0:20000]:
    if (line.strip() == ''): continue
    input_text, target_text, _ = line.split('\t')
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text.lower())
    target_texts.append(target_text.lower())

t1=Tokenizer()
t1.fit_on_texts(input_texts)
encoded_docs=t1.texts_to_sequences(input_texts)
encoder_seq_length = max([len(x) for x in encoded_docs])
encoder_input_data = pad_sequences(encoded_docs, maxlen=encoder_seq_length, padding='post')

t2=Tokenizer(char_level=True)
t2.fit_on_texts(target_texts)
encoded_docs=t2.texts_to_sequences(target_texts)
decoder_seq_length = max([len(x) for x in encoded_docs])
decoder_input_data = pad_sequences(encoded_docs, maxlen=decoder_seq_length, padding='post')

num_encoder_tokens = len(t1.word_counts)
num_decoder_tokens = len(t2.word_counts)

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', encoder_seq_length)
print('Max sequence length for outputs:', decoder_seq_length)

input_token_index = t1.word_index
target_token_index = t2.word_index
decoder_target_data = np.zeros(
    (len(input_texts), decoder_seq_length, num_decoder_tokens+1),
    dtype='float32')

for i, target_text in enumerate(target_texts):
    for t, char in enumerate(target_text):
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.

# 2. build model
# 多GPU
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    encoder_inputs = Input(shape=(encoder_seq_length,))
    encoder_embed = Embedding(vocab_size, embed_size)
    encoder_embed_output = encoder_embed(encoder_inputs)
    encoder = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_embed_output)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(decoder_seq_length,))
    decoder_embed = Embedding(num_decoder_tokens+1, embed_size)
    decoder_embed_output = decoder_embed(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embed_output,
                                         initial_state=encoder_states)
    # 将encoder的state_h就是context c和decoder rnn的输出拼接
    expand_h = tf.expand_dims(state_h,axis=1)
    expand_h_list = []
    for _ in range(decoder_outputs.shape[1]):
        expand_h_list.append(expand_h)
    concat_expand_h = Concatenate(axis=1)(expand_h_list )
    # Ohh't+Oye(yt-1)+Occ
    concat_decoder_outputs = Concatenate(axis=2)([decoder_outputs,concat_expand_h, decoder_embed_output])

    decoder_output_layer = Sequential([
        Dense(512, activation='relu'),  # st = relu(Ohh't+Oye(yt-1)+Occ)
        MaxPooling1D(pool_size=2,       # max-out
                     strides=2,
                     data_format='channels_first',
                     padding='valid'),
        Dense(num_decoder_tokens + 1, activation='softmax')
    ])
    model_outputs = decoder_output_layer(concat_decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], model_outputs)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])

# exit(0)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs)
model.save('s2s.h5')

# 3. buiding the inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_input_context = Input(shape=(latent_dim,))
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embed_output, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

# 将c和decoder的输出拼接
expand_c = tf.expand_dims(decoder_input_context,axis=1)
# 因为在推断阶段是逐个字符解码，因此decoder的输出只有一个字符。此处提前出的第一个位置的decoder的输出
concat_decoder_outputs = Concatenate(axis=2)([decoder_outputs[:,:1,:],
                                              expand_c,
                                              decoder_embed_output[:,:1,:]
                                              ])

decoder_outputs2 = decoder_output_layer(concat_decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs+[decoder_input_context],
    [decoder_outputs2] + decoder_states, name="infer_model")

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq, verbose=0)
    context_c = states_value[0]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['\t']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value+[context_c], verbose=0)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
                len(decoded_sentence) > decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence

import random
size = encoder_input_data.shape[0]
for _ in range(100):
    idx = random.randint(0, size)
    input_seq = encoder_input_data[idx: idx + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input sentence:', input_texts[idx])
    print('Decoded sentence:', decoded_sentence)

