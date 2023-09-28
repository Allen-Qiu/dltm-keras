"""
该文件读入保存的模型参数和文本处理器TextVectorization
然后在模型上inference
@author: Allen Qiu
"""
import tensorflow as tf
from encoder_decoder_translation_attention_models import Encoder, Decoder, CrossAttention, Translator
import numpy as np
import pickle
from tensorflow.keras.layers import TextVectorization

# 1. 读入待翻译的文本
data_path = '../data/cmn.txt'
source_texts_raw = []
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[0:20000]:
    if (line.strip() == ''): continue
    input_text, target_text, _ = line.split('\t')
    source_texts_raw.append(input_text.lower())

# Load TextVectorization
s_model = tf.keras.models.load_model("source_text_processor.model")
source_text_processor = s_model.layers[0]
t_model = tf.keras.models.load_model("target_text_processor.model")
target_text_processor = t_model.layers[0]

model = Translator(source_text_processor, target_text_processor)
model.load_weights('./checkpoints/my_checkpoint')
idx = np.random.choice(np.arange(len(source_texts_raw)), size=100)
for i in idx:
    print(f"source: {source_texts_raw[i]}")
    result = model.translate([source_texts_raw[i]])
    s = result[0].numpy().decode()
    print(f"target: {s}")