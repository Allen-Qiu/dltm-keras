import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from transformers import TFAutoModel

fin = open('yelp-reviews-small.json')
lines = fin.readlines()
reviews = list()
stars = list()

for line in lines:
    rdict = json.loads(line)
    reviews.append(rdict['text'])
    stars.append(rdict['stars'])

local = r"C:\Users\qjt16\Desktop\bert-base-uncased"
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(local)
classifier = TFAutoModel.from_pretrained(local)

inputs = tokenizer(reviews, padding=True, max_length=512, truncation=True, return_tensors="tf")
outputs = classifier(inputs)
predictions = tf.math.softmax(outputs.logits, axis=-1)


k = 0
for review, star in zip(reviews,stars):
    try:
        r=classifier(
            review
        )
        print("predict:%s, label:%s"%(r[0]['label'], star))
        if k>100: break
        k += 1
    except:
        None
#%%
# import tensorflow as tf
# from transformers import AutoTokenizer
# from transformers import TFAutoModel
# from transformers import TFAutoModelForSequenceClassification

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# checkpoint = r"C:\Users\qjt16\Desktop\bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# classifier = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

# inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="tf", max_length=512)
# outputs = classifier(inputs)
# predictions = tf.math.softmax(outputs.logits, axis=-1)