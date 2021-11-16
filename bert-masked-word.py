'''
predict masked word using bert
'''

from transformers import BertTokenizer
from transformers import TFBertForMaskedLM, TFBertModel
import tensorflow as tf

checkpoint = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(checkpoint)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = TFBertForMaskedLM.from_pretrained(checkpoint)

text = "his mother is a [MASK]."
tokenized_text = tokenizer.tokenize(text)
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="tf")
masked_index = tokenized_text.index("[MASK]")
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
outputs = model(inputs)
predictions = outputs[0]

m = predictions[0,masked_index+1,:]
out = tf.keras.backend.softmax(m)
r = tf.math.top_k(out, k=5)
idx = r.indices.numpy()
for i in idx:
    word = tokenizer.convert_ids_to_tokens([i])[0]
    print(word)



