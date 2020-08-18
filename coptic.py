import argparse
from os.path import join as j
import sys

import stanza.models.parser as parser


def train(args):
    args['wordvec_dir'] = j("stanza_data", "wordvec")
    #args['train_file'] = j("stanza_data", "scriptorium", "cs-ud-train-preprocessed.conllu")
    #args['eval_file'] = j("stanza_data", "scriptorium", "cs-ud-dev-preprocessed.conllu")
    #args['gold_file'] = j("stanza_data", "scriptorium", "cs-ud-dev.conllu")
    args['train_file'] = j("stanza_data", "scriptorium", "cs-ud-train-and-dev-preprocessed.conllu")
    args['eval_file'] = j("stanza_data", "scriptorium", "cs-ud-minitest-preprocessed.conllu")
    args['gold_file'] = j("stanza_data", "scriptorium", "cs-ud-minitest.conllu")
    args['output_file'] = j('stanza_data', "scriptorium", "cs-ud-minitest-pred.conllu")
    args['mode'] = 'train'
    parser.train(args)


def test(args):
    args['mode'] = "predict"
    args['eval_file'] = j("stanza_data", "scriptorium", "cs-ud-test-preprocessed.conllu")
    args['gold_file'] = j("stanza_data", "scriptorium", "cs-ud-test.conllu")
    args['output_file'] = j('stanza_data', "scriptorium", "cs-ud-test-pred.conllu")
    return parser.evaluate(args)


def trial(args):
    train(args)
    las = test(args)
    return las


def search(args):
    # most trials seem to converge by 6000
    args['max_steps'] = 6000

    from hyperopt import hp, fmin, Trials, STATUS_OK, tpe
    from hyperopt.pyll import scope

    # params to search for
    space = {
        'optim': hp.choice('optim', ['sgd', 'adagrad', 'adam', 'adamax']),
        'hidden_dim': scope.int(hp.quniform('hidden_dim', 150, 400, 50)),
    }
    # f to minimize
    def f(opted_args):
        new_args = args.copy()
        new_args.update(opted_args)
        print("Trial with args:", opted_args)
        return {'loss': 1 - trial(new_args), 'status': STATUS_OK}

    trials = Trials()
    best = fmin(f, space, algo=tpe.suggest, max_evals=200, trials=trials)
    print("\nBest parameters:\n" + 30 * "=")
    print(best)

    trials = [t for t in trials]
    print("\n\nRaw trial output")
    for tt in trials:
        print(tt)

    print("\n\n")

    print("\nTrials:\n")
    for i, tt in enumerate(trials):
        if i == 0:
            print("LAS\t" + "\t".join(list(tt['misc']['vals'].keys())))
        vals = map(lambda x:str(x[0]), tt['misc']['vals'].values())
        las = str(1-tt['result']['loss'])
        print('\t'.join([las, "\t".join(vals)]))


def main(mode):
    args = vars(parser.parse_args())
    args['lang'] = "cop"
    args['shorthand'] = "cop_scriptorium"
    args['treebank'] = "cop_scriptorium"
    args['tag_type'] = "gold"

    # parameters we're pretty sure about
    args['word_emb_dim'] = 50
    args['char_emb_dim'] = 50
    args['tag_emb_dim'] = 5
    args['batch_size'] = 1500
    args['num_layers'] = 3
    args['word_dropout'] = 0.3
    args['dropout'] = 0.5
    args['char_hidden_dim'] = 250
    args['deep_biaff_hidden_dim'] = 200
    args['hidden_dim'] = 300

    if mode == 'train':
        train(args.copy())
    elif mode == 'test':
        test(args.copy())
    else:
        search(args)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=['hyperopt', 'train', 'test'])
    args = ap.parse_args()
    sys.argv.pop()
    main(args.mode)


    """set -o xtrace
echo "Running parser with $args..."
python -m stanza.models.parser \
        --wordvec_dir $WORDVEC_DIR \
        --train_file $train_file \
        --eval_file $eval_file \
        --output_file $output_file \
        --gold_file $gold_file \
        --lang $lang \
        --shorthand $short \
        --batch_size $batch_size \
        --mode train \
        $args
        """
