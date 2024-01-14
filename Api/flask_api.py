# -*- coding: UTF-8 -*-
import time
import os
import json
import base64
import torch
import random
import sys
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from flask_cors import CORS
from flask import Flask, request, jsonify
from prediction import predict

ROOT = os.path.abspath(os.path.dirname(os.getcwd()))
print(ROOT)
sent_delimiter = ['，', '。', '？', '！', '；']

# print(f'using {device} device')
app = Flask(__name__)
CORS(app)
time_start = 0

electra_tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-180g-base-discriminator')
ElectraClassifier = ElectraForSequenceClassification.from_pretrained('hfl/chinese-electra-180g-base-discriminator')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if sys.platform == 'darwin': device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)

ElectraClassifier = torch.load(
    os.path.join('useModel', 'ELECTRA_binary_seq_cnv0.8.1_20.h5'),
    map_location=device)

ElectraClassifier.to(device)

ElectraClassifier.eval()

correct = []
error = []

with open(os.path.join('static', 'example.txt'), 'r', encoding='utf-8') as f:
    data_ = f.readlines()
    for line in data_:
        line = line.split("^")
        if '0' in line[1]:
            correct.append(line[0])
        else:
            error.append(line[0])


@app.route('/test', methods=['GET'])
def getResult():
    print('hey hey')
    return 'hey hey'


@app.route('/predict', methods=['GET', 'POST'])
def postInput():
    insertValues = bytes(request.args['q'], 'utf-8')
    insertValues = str(json.loads(insertValues))

    result = {'chunks': [insertValues], 'chinese': insertValues}
    result = Input(result)

    return result


def module_setting(data):
    encoded_sent = electra_tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=data,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True
    )

    # Convert lists to tensors
    input_ids = torch.tensor(encoded_sent['input_ids'])
    attention_masks = torch.tensor(encoded_sent['attention_mask'])

    return input_ids, attention_masks


def get_test_dataloader(inputs, attention_mask):
    expand_dim = lambda tlist, x: [tlist[i:i + x] for i in range(0, len(tlist), x)]
    inputs = expand_dim(inputs, 64)
    attention_mask = expand_dim(attention_mask, 64)
    dataset = []
    for i, k in zip(inputs, attention_mask):
        dataset.append(tuple((i, k)))
    return dataset


def Input(insertValues):
    sentences = insertValues['chunks']
    chinese = insertValues['chinese']
    result = []
    num = 0
    sentences = list(map(lambda x: x + '。' if x[-1] not in sent_delimiter else x, sentences))  # 如果最後一個字不是標點符號，就加上句號
    test_inputs, test_attention_mask = module_setting(sentences)
    test_dataloader = get_test_dataloader(test_inputs, test_attention_mask)
    for i, batch in enumerate(test_dataloader):
        id, sentence, predict_, num = predict(batch, ElectraClassifier, electra_tokenizer, num, device)
        result.append(id)
        result.append(sentence)
        result.append(predict_)

    output = []
    for i, j, k in zip(result[0], result[1], result[2]):
        output.append([i, j, k])

    return output[0][0]


if __name__ == "__main__":
    app.run(port=8080, debug=True)
