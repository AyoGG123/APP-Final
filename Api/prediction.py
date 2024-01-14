# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
import re


def predict(batch, bert_classifier, electra_tokenizer, num, device):
    output = {'id': [], 'chunks': [], 'pred': []}
    sent_delimiter_ids = [8024, 511, 8043, 8013, 8039]
    # ['，', '。', '？', '！', '；']

    for input_ids, attention_mask in zip(*tuple(t.to(device) for t in batch)):
        s_input_ids, s_attention_mask = split_chunk_into_sent(input_ids, attention_mask, sent_delimiter_ids)

        s_input_ids = s_input_ids.to(device)
        s_attention_mask = s_attention_mask.to(device)

        with torch.no_grad():
            logits = bert_classifier(input_ids=s_input_ids, attention_mask=s_attention_mask)

        pred_result = torch.argmax(logits[0], dim=1).tolist()
        t = electra_tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
        num += 1
        temp = re.sub(r' ', '', t)

        for line in re.split("，|。|？|！|；", temp):
            output['chunks'].append(line)

        output['chunks'] = output['chunks'][:-1]
        output['pred'] = pred_result

        for i in range(len(output['pred'])):
            output['id'].append(str(i + 1))

        for i in range(len(output['pred'])):
            # print(output['pred'][i])
            # print(type(output['pred'][i]))
            output['pred'][i] = '錯誤' if output['pred'][i] == 1 else '正確'

    torch.cuda.empty_cache()
    return output['id'], output['chunks'], output['pred'], num


'''def predict(batch, bert_classifier, electra_tokenizer, num, device):
    result = {'ID': [], '句子': [], '預測': []}

    b_input_ids, b_attention_mask = batch
    b_input_ids = b_input_ids.to(device)
    b_attention_mask = b_attention_mask.to(device)
    with torch.no_grad():
        output = bert_classifier(input_ids=b_input_ids, attention_mask=b_attention_mask)
        logits = output.logits
    text = electra_tokenizer.batch_decode(b_input_ids, skip_special_tokens=True)
    text = [list(filter(lambda x: x != '[PAD]', x.split(' '))) for x in text]
    for i in text:
        if '[CLS]' in i: i.remove('[CLS]')
        if '[SEP]' in i: i.remove('[SEP]')

    for logit, t, _ids in zip(logits, text, b_input_ids):
        pred = logit.argmax().item()
        # threshold = 0.97
        # pred = 1 if prob[1]>threshold else 0
        num += 1
        result['ID'].append(str(num))
        result['句子'].append(''.join(t))
        if str(pred) == '0':
            result['預測'].append('沒有文法錯誤')
        else:
            result['預測'].append('有文法錯誤')
        # result['預測'].append(str(pred))

    return str(result['ID'])[1:-1], str(result['句子'])[1:-1], str(result['預測'])[1:-1], num'''


# 分割標點符號
def split_chunk_into_sent(input_ids, attention_mask, sent_delimiter_ids):
    s_input_ids = []
    s_attention_mask = []
    start = 0
    assert len(input_ids) == len(attention_mask)

    for inx, (id, att) in enumerate(zip(input_ids, attention_mask)):
        if att == 0: break
        if id in sent_delimiter_ids:
            input_id = [101] + input_ids[start:inx + 1].tolist() + [102] + [0 for _ in range(
                512 - 2 - len(input_ids[start:inx + 1]))]
            input_id = torch.tensor(input_id)
            s_input_ids.append(input_id)
            s_attention_mask.append(torch.tensor([1] + attention_mask[start:inx + 1].tolist() + [1] + [0 for _ in range(
                512 - 2 - len(attention_mask[start:inx + 1]))]))
            start = inx + 1

    return torch.stack(s_input_ids), torch.stack(s_attention_mask)


if __name__ == "__main__":
    pass
