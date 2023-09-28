"""
加入注意力机制的 english-chinese翻译模型
"""
import tensorflow as tf

# 1. Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
                                                   mask_zero=True)
        self.rnn = tf.keras.layers.LSTM(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
    def call(self, x):
        x = self.embedding(x)
        x, state_h, state_c = self.rnn(x)
        return x, state_h, state_c

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
          texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context, state_h, state_c = self(context)
        return context, state_h, state_c

# 2. attention layer
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):            # x是decoder的RNN输出， context是Encoder的RNN输出
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)    # 当前没有给出key, 所以使用value当作key

        x = self.add([x, attn_output])       # 相当于Decoder的输出和计算注意力后的结果再做拼接
        x = self.layernorm(x)
        return x

# 3. decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.text_processor = text_processor
        self.word_to_id = tf.keras.layers.StringLookup(
            mask_token='',
            vocabulary=text_processor.get_vocabulary())
        self.id_to_word = tf.keras.layers.StringLookup(
            mask_token='',
            vocabulary=text_processor.get_vocabulary(),
            invert=True)
        self.start_token = self.word_to_id('\t')
        self.end_token = self.word_to_id('\n')
        self.units = units
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   units, mask_zero=True)
        self.rnn = tf.keras.layers.LSTM(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.attention = CrossAttention(units)
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self,
             context, x,        # context是encoder的RNN输出，x是decoder的RNN输入
             state=None,
             return_state=False):
        x = self.embedding(x)
        x, state_h, state_c = self.rnn(x, initial_state=state)
        x = self.attention(x, context)
        logits = self.output_layer(x)

        if return_state:
            return logits, [state_h, state_c]
        else:
            return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        return start_tokens, done

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result

    def get_next_token(self, context, next_token, done, state, temperature=0.0):
        logits, state = self(
            context, next_token,
            state=state,
            return_state=True)

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            next_token = tf.random.categorical(logits, num_samples=1)
        done = done | (next_token == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state

# 4. 翻译模型
class Translator(tf.keras.Model):
  def __init__(self,
               source_text_processor,
               target_text_processor,
               units=256):
    super().__init__()
    self.encoder = Encoder(source_text_processor, units)
    self.decoder = Decoder(target_text_processor, units)

  def call(self, inputs):
    source, x = inputs
    context,state_h,state_c = self.encoder(source)
    logits = self.decoder(context, x, [state_h, state_c])
    return logits

  def translate(self,
                texts, *,
                max_length=50,
                temperature=0.0):
      context, state_h, state_c = self.encoder.convert_input(texts)
      tokens = []
      state= [state_h, state_c]
      next_token, done = self.decoder.get_initial_state(context)

      for _ in range(max_length):
          next_token, done, state = self.decoder.get_next_token(
              context, next_token, done, state, temperature)

          tokens.append(next_token)
          if tf.executing_eagerly() and tf.reduce_all(done):
              break
      tokens = tf.concat(tokens, axis=-1)  # t*[(batch 1)] -> (batch, t)
      result = self.decoder.tokens_to_text(tokens)
      return result
