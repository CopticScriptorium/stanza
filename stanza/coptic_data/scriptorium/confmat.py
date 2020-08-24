import conllu
from sklearn.metrics import confusion_matrix

gold = conllu.parse(open('cs-ud-test.conllu', 'r').read())
pred = conllu.parse(open('cs-ud-test-pred.conllu', 'r').read())

gold_labels = [t['deprel'] for sent in gold for t in sent]
pred_labels = [t['deprel'] for sent in pred for t in sent]
vocab = sorted(list(set(gold_labels + pred_labels)))

print("\t".join(['_'] + vocab))
for i,line in enumerate(confusion_matrix(gold_labels, pred_labels, labels=vocab)):
    print("\t".join([vocab[i]] + [n if int(n) > 0 else "" for n in list(map(str, list(line)))]))
