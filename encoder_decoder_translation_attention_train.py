"""
加入注意力机制的 english-chinese翻译模型。
改编自https://www.tensorflow.org/text/tutorials/nmt_with_attention
当前文件训练一个模型，并把参数存储到磁盘
@author: Allen Qiu
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from encoder_decoder_translation_attention_models import Translator

lr = 0.0008          # learning rate
epochs = 100         # Number of epochs to train for.
UNITS = 256         # hidden units
embed_size = 128
data_path = '../data/cmn.txt'

# 1. build dataset
source_texts_raw = []
target_texts_raw = []
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[0:20000]:
    if (line.strip() == ''): continue
    input_text, target_text, _ = line.split('\t')
    target_text = '\t' + target_text + '\n'
    source_texts_raw.append(input_text.lower())
    target_texts_raw.append(target_text.lower())

source_texts = np.array(source_texts_raw)
target_texts = np.array(target_texts_raw)

## 1.1 building tf.data.Dataset
BUFFER_SIZE = len(source_texts)
BATCH_SIZE  = 64

train_texts = (
    tf.data.Dataset
    .from_tensor_slices((source_texts, target_texts))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))

## 1.2. 分别为源文本和目标文本创建TextVectorization工具
source_text_processor = tf.keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    ragged=True)
source_text_processor.adapt(source_texts)

target_text_processor = tf.keras.layers.TextVectorization(
    standardize="lower",
    split="character",                      # 中文在字符级上进行翻译
    ragged=True)
target_text_processor.adapt(target_texts)

## 1.3 建立最终的训练集
def process_text(source, target):
  source = source_text_processor(source).to_tensor()
  target = target_text_processor(target)
  targ_in = target[:,:-1].to_tensor()
  targ_out = target[:,1:].to_tensor()
  return (source, targ_in), targ_out

train_ds = train_texts.map(process_text, tf.data.AUTOTUNE)

# 2. 训练模型
def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

model = Translator(source_text_processor, target_text_processor, units=UNITS)
opt = Adam(learning_rate=lr)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
model.compile(optimizer=opt,  loss=masked_loss,  metrics=[masked_acc])

model.fit(train_ds, epochs=epochs)
model.save_weights('./checkpoints/my_checkpoint')

# save TextVectorization
# 因为TextVectorization层不能直接保存到磁盘，这里用了点小技巧。
# 即创建一个模型，把TextVectorization作为模型的一个层
s_model = tf.keras.models.Sequential()
s_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
s_model.add(source_text_processor)
s_model.save("source_text_processor.model", save_format="tf")

t_model = tf.keras.models.Sequential()
t_model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
t_model.add(target_text_processor)
t_model.save("target_text_processor.model", save_format="tf")
