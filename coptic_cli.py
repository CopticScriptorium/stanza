import argparse
import sys
from os.path import join as j
from stanza.coptic import train, test, Predictor, _hyperparam_search


def main(args):
    mode = args.mode
    if mode == 'train':
        train(
            j("stanza/coptic_data", "scriptorium", "cs-ud-train-and-dev.conllu"),
            j("stanza/coptic_data", "scriptorium", "cs-ud-minitest.conllu")
        )
        # train(
        #    j("coptic_data", "scriptorium", "cs-ud-train-preprocessed.conllu"),
        #    j("coptic_data", "scriptorium", "cs-ud-dev-preprocessed.conllu")
        # )
    elif mode == 'test':
        test(j("stanza/coptic_data", "scriptorium", "cs-ud-test.conllu"))
    elif mode == 'predict':
        p = Predictor()
        sys.stdout.buffer.write(p.predict(args.pred_file).encode("utf8"))
    else:
        _hyperparam_search()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=['hyperopt', 'train', 'test', 'predict'])
    ap.add_argument("--pred-file", type=str)
    args = ap.parse_args()
    sys.argv.pop()
    main(args)
