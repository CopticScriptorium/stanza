import conllu
from sklearn.metrics import confusion_matrix

gold = conllu.parse(open('cs-ud-test.conllu', 'r').read())
pred = conllu.parse(open('cs-ud-test-pred.conllu', 'r').read())

gold_labels = [t['deprel'] for sent in gold for t in sent]
pred_labels = [t['deprel'] for sent in pred for t in sent]
vocab = sorted(list(set(gold_labels + pred_labels)))

import sys
l1 = sys.argv[1]
l2 = sys.argv[2]

print("sent_id\ttok_num\tdeprel_gold\tdeprel_pred\tsent")
for g_sent, p_sent in zip(gold, pred):
    for i, (gt, pt) in enumerate(zip(g_sent, p_sent)):
        if gt['deprel'] == l1 and pt['deprel'] == l2 \
                or gt['deprel'] == l2 and pt['deprel'] == l1:
            line = []
            line.append(g_sent.metadata['sent_id'])
            line.append(str(i))
            line.append(gt['deprel'])
            line.append(pt['deprel'])
            line.append(" ".join(t['form'] if i != j else '>>' + t['form'] + '<<' 
                                 for j,t in enumerate(g_sent) 
                                 if type(t['id']) == int))
            print('\t'.join(line))

