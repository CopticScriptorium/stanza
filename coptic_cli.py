import argparse
import sys
from os.path import join as j
from stanza.coptic import train, test, _hyperparam_search


def main(mode):
    if mode == 'train':
        train(
            j("stanza_data", "scriptorium", "cs-ud-train-and-dev.conllu"),
            j("stanza_data", "scriptorium", "cs-ud-minitest.conllu")
        )
        # train(
        #    j("stanza_data", "scriptorium", "cs-ud-train-preprocessed.conllu"),
        #    j("stanza_data", "scriptorium", "cs-ud-dev-preprocessed.conllu")
        # )
    elif mode == 'test':
        test(j("stanza_data", "scriptorium", "cs-ud-test.conllu"))
    else:
        _hyperparam_search()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=['hyperopt', 'train', 'test'])
    args = ap.parse_args()
    sys.argv.pop()
    main(args.mode)
