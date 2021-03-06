import argparse
import random, os
from os.path import join as j
from collections import OrderedDict
import conllu
import torch
import pathlib
import tempfile
import depedit

import stanza.models.parser as parser
from stanza.models.depparse.data import DataLoader
from stanza.models.depparse.trainer import Trainer
from stanza.models.common import utils
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.doc import *
from stanza.utils.conll import CoNLL

PACKAGE_BASE_DIR = pathlib.Path(__file__).parent.absolute()

# Parser arguments -----------------------------------------------------------------------------------------------------
# These args are used by stanza.models.parser. Keys should always exactly match those you'd get from the dictionary
# obtained from running stanza.models.parser.parse_args(). The values below were selected through hyperoptimization.
DEFAULT_PARSER_ARGS = {
    # general setup
    'lang': 'cop',
    'treebank': 'cop_scriptorium',
    'shorthand': 'cop_scriptorium',
    'data_dir': j(PACKAGE_BASE_DIR, 'data', 'depparse'),
    'output_file': j(PACKAGE_BASE_DIR, 'coptic_data', 'scriptorium', 'pred.conllu'),
    'seed': 1234,
    'cuda': torch.cuda.is_available(),
    'cpu': not torch.cuda.is_available(),
    'save_dir': j(PACKAGE_BASE_DIR, "..", 'stanza_models'),
    'save_name': None,

    # word embeddings
    'pretrain': True,
    'wordvec_dir': j(PACKAGE_BASE_DIR, 'coptic_data', 'wordvec'),
    'wordvec_file': j(PACKAGE_BASE_DIR, 'coptic_data', 'wordvec', 'word2vec', 'Coptic', 'coptic_50d.vec.xz'),
    'word_emb_dim': 50,
    'word_dropout': 0.3,

    # char embeddings
    'char': True,
    'char_hidden_dim': 200,
    'char_emb_dim': 50,
    'char_num_layers': 1,
    'char_rec_dropout': 0,  # very slow!

    # pos tags
    'tag_emb_dim': 5,
    'tag_type': 'gold',

    # network params
    'hidden_dim': 300,
    'deep_biaff_hidden_dim': 200,
    'composite_deep_biaff_hidden_dim': 100,
    'transformed_dim': 75,
    'num_layers': 3,
    'pretrain_max_vocab': 250000,
    'dropout': 0.5,
    'rec_dropout': 0,  # very slow!
    'linearization': True,
    'distance': True,

    # training
    'sample_train': 1.0,
    'optim': 'adam',
    'lr': 0.002,
    'beta2': 0.95,
    'max_steps': 20000,
    'eval_interval': 100,
    'max_steps_before_stop': 2000,
    'batch_size': 1500,
    'max_grad_norm': 1.0,
    'log_step': 20,

    # these need to be included or there will be an error when stanza tries to access them
    'train_file': None,
    'eval_file': None,
    'gold_file': None,
    'mode': None,
}

# Custom features ------------------------------------------------------------------------------------------------------
# Params for controlling the custom features we're feeding the network
FEATURE_CONFIG = {
    # BIOLU or BIO
    'features': [
        'foreign_word',
        'morph_count',
        'left_morph',
        'entity',
    ],
    'foreign_word_binary': True,
    'morph_count_binary': False,
    'entity_encoding_scheme': 'BIOLU',
    'entity_dropout': 0.30,
}

# DepEdit preprocessor which removes gold morph data and makes a few other tweaks
PREPROCESSOR = depedit.DepEdit(config_file=j(PACKAGE_BASE_DIR, "coptic_data", "depedit", "add_ud_and_flat_morph.ini"),
                               options=type('', (), {"quiet": True, "kill": "both"}))

# Load a lexicon of foreign words and initialize a lemma cache
with open(j(PACKAGE_BASE_DIR, 'coptic_data', 'lang_lexicon.tab'), 'r', encoding="utf8") as f:
    FOREIGN_WORDS = {x.split('\t')[0]: x.split('\t')[1].rstrip()
                     for x in f.readlines() if '\t' in x}
FW_CACHE = {}

# load known entities and sort in order of increasing token length
with open(j(PACKAGE_BASE_DIR, 'coptic_data', 'entities.tab'), 'r', encoding="utf8") as f:
    KNOWN_ENTITIES = OrderedDict(sorted(
        ((x.split('\t')[0], x.split('\t')[1]) for x in f.readlines()),
        key=lambda x: len(x[0].split(" "))
    ))


def _add_entity_feature(feature_config, sentences, predict=False):
    # unless we're predicting, use dropout to pretend we don't know some entities
    dropout_entities = {
        estr: etype for estr, etype in KNOWN_ENTITIES.items()
        # three ways for an entity to not get dropped out:
        # 1. we're predicting (all tokens stay)
        # 2. it has only one token
        # 3. we roll above the dropout threshold
        if (predict
            or (' ' not in estr)
            or (random.random() >= feature_config['entity_dropout']))
    }

    def find_span_matches(tokens, pattern):
        slen = len(pattern)
        matches = []
        for i in range(len(tokens) - (slen - 1)):
            if tokens[i:i + slen] == pattern:
                matches.append((i, slen))
        return matches

    def delete_conflicting(new_span, entities, entity_tags):
        overlap_exists = lambda range1, range2: set(range1).intersection(range2)
        span = lambda begin, length: list(range(begin, begin + length))

        new_span = span(*new_span)
        for i in range(len(entities) - 1, -1, -1):
            begin, length, _ = entities[i]

            # in case of overlap, remove the old entity and pop it off the list
            old_span = span(begin, length)
            if overlap_exists(new_span, old_span):
                for j in old_span:
                    entity_tags[j] = "O"
                entities.pop(i)

    def encode(new_span, entity_tags, entity_type):
        assert feature_config['entity_encoding_scheme'] in ["BIOLU", "BIO"]
        if feature_config['entity_encoding_scheme'] == "BIOLU":
            unit_tag = "U-"
            begin_tag = "B-"
            inside_tag = "I-"
            last_tag = "L-"
        else:
            unit_tag = "B-"
            begin_tag = "B-"
            inside_tag = "I-"
            last_tag = "I-"

        begin, length = new_span
        if length == 1:
            entity_tags[begin] = unit_tag + entity_type
        else:
            for i in range(begin, begin + length):
                if i == begin:
                    entity_tags[i] = begin_tag + entity_type
                elif i == (begin + length - 1):
                    entity_tags[i] = last_tag + entity_type
                else:
                    entity_tags[i] = inside_tag + entity_type

    # use BIOLU encoding for entities https://github.com/taasmoe/BIO-to-BIOLU
    # in case of nesting, longer entity wins
    for sentence in sentences:
        tokens = [t['form'] for t in sentence]
        entity_tags = (['O'] * len(tokens))

        entities = []
        for entity_string, entity_type in dropout_entities.items():
            new_spans = find_span_matches(tokens, entity_string.split(" "))
            for new_span in new_spans:
                delete_conflicting(new_span, entities, entity_tags)
                encode(new_span, entity_tags, entity_type)
                entities.append((new_span[0], new_span[1], entity_type))

        for token, entity_tag in zip(sentence, entity_tags):
            token['feats']['Entity'] = entity_tag


def _add_morph_count_feature(feature_config, sentences, predict=False):
    for sentence in sentences:
        for token in sentence:
            feats = token['feats']
            misc = token['misc']

            feats['MorphCount'] = (
                '1' if misc is None or 'Morphs' not in misc
                else (
                    'Many' if feature_config['morph_count_binary']
                    else str(len(misc['Morphs'].split('-')))
                )
            )
            token['feats'] = feats
    return sentences


def _add_left_morph_feature(feature_config, sentences, predict=False):
    for sentence in sentences:
        for token in sentence:
            feats = token['feats']
            misc = token['misc']
            if misc is not None and 'Morphs' in misc:
                feats['LeftMorph'] = misc['Morphs'].split('-')[0]
            token['feats'] = feats
    return sentences


def _add_foreign_word_feature(feature_config, sentences, predict=False):
    def foreign_word_lookup(lemma):
        if lemma in FW_CACHE:
            return FW_CACHE[lemma]

        for fw, lang in FOREIGN_WORDS.items():
            glob_start = fw[0] == '*'
            glob_end = fw[-1] == '*'
            fw = fw.replace('*', '')
            if glob_start and glob_end and fw in lemma:
                FW_CACHE[lemma] = lang
                return lang
            elif glob_start and lemma.endswith(fw):
                FW_CACHE[lemma] = lang
                return lang
            elif glob_end and lemma.startswith(fw):
                FW_CACHE[lemma] = lang
                return lang
            elif lemma == fw:
                FW_CACHE[lemma] =lang
                return lang
        FW_CACHE[lemma] = False
        return False

    for sentence in sentences:
        for token in sentence:
            feats = token['feats']
            lang_of_origin = foreign_word_lookup(token['lemma'])
            feats['ForeignWord'] = (
                'No' if not lang_of_origin
                else (
                    'Yes' if feature_config['foreign_word_binary']
                    else lang_of_origin
                ))
            token['feats'] = feats
    return sentences


FEATURE_FUNCTIONS = {
    'foreign_word': _add_foreign_word_feature,
    'left_morph': _add_left_morph_feature,
    'morph_count': _add_morph_count_feature,
    'entity': _add_entity_feature,
}


def _preprocess(feature_config, conllu_string, predict):
    # remove gold information
    s = PREPROCESSOR.run_depedit(conllu_string)

    # deserialize so we can add custom features
    sentences = conllu.parse(s)
    for sentence in sentences:
        for token in sentence:
            if token['feats'] is None:
                token['feats'] = OrderedDict()
    for feature_name in feature_config['features']:
        assert feature_name in FEATURE_FUNCTIONS.keys()
        FEATURE_FUNCTIONS[feature_name](feature_config, sentences, predict=predict)

    # serialize and return
    return "".join([sentence.serialize() for sentence in sentences])


def _read_conllu_arg(conllu_filepath_or_string, feature_config, gold=False, predict=False):
    try:
        conllu.parse(conllu_filepath_or_string)
        s = conllu_filepath_or_string
    except:
        try:
            with open(conllu_filepath_or_string, 'r', encoding="utf8") as f:
                s = f.read()
                conllu.parse(s)
        except:
            raise Exception(f'"{conllu_filepath_or_string}" must either be a valid conllu string '
                            f'or a filepath to a valid conllu string')

    if not gold:
        s = _preprocess(feature_config, s, predict)

    tempf = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
    tempf.write(s)
    tempf.close()
    return tempf.name


# public api -----------------------------------------------------------------------------------------------------------
def train(train, dev, save_name=None):
    """Train a new stanza model.

    :param train: either a conllu string or a path to a conllu file
    :param dev: either a conllu string or a path to a conllu file
    :param save_name: optional, a name for your model's save file, which will appear in 'stanza_models/'
    """
    args = DEFAULT_PARSER_ARGS.copy()
    feature_config = FEATURE_CONFIG.copy()
    args['mode'] = 'train'
    args['train_file'] = _read_conllu_arg(train, feature_config)
    args['eval_file'] = _read_conllu_arg(dev, feature_config)
    args['gold_file'] = _read_conllu_arg(dev, feature_config, gold=True)
    if save_name:
        args['save_name'] = save_name
    parser.train(args)


def test(test, save_name=None):
    """Evaluate using an existing stanza model.

    :param test: either a conllu string or a path to a conllu file
    :param save_name: optional, a name for your model's save file, which will appear in 'stanza_models/'
    """
    args = DEFAULT_PARSER_ARGS.copy()
    feature_config = FEATURE_CONFIG.copy()
    args['mode'] = "predict"
    args['eval_file'] = _read_conllu_arg(test, feature_config)
    args['gold_file'] = _read_conllu_arg(test, feature_config, gold=True)
    if save_name:
        args['save_name'] = save_name
    return parser.evaluate(args)


class Predictor:
    """Wrapper class so model can sit in memory for multiple predictions"""

    def __init__(self, args=None, feature_config=None):
        if args is None:
            args = DEFAULT_PARSER_ARGS.copy()
        if feature_config is None:
            self.feature_config = FEATURE_CONFIG.copy()
        model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
            else '{}/{}_parser.pt'.format(args['save_dir'], args['shorthand'])

        # load pretrain; note that we allow the pretrain_file to be non-existent
        pretrain_file = '{}/{}.pretrain.pt'.format(args['save_dir'], args['shorthand'])
        self.pretrain = Pretrain(pretrain_file)

        # load model
        print("Loading model from: {}".format(model_file))
        use_cuda = args['cuda'] and not args['cpu']
        self.trainer = Trainer(pretrain=self.pretrain, model_file=model_file, use_cuda=use_cuda)
        self.loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
        self.batch_size = args['batch_size']

        # load config
        for k in args:
            if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
                self.loaded_args[k] = args[k]

    def predict(self, eval_file_or_string):
        eval_file = _read_conllu_arg(eval_file_or_string, self.feature_config, predict=True)
        doc = Document(CoNLL.conll2dict(input_file=eval_file))
        batch = DataLoader(
            doc,
            self.batch_size,
            self.loaded_args,
            self.pretrain,
            vocab=self.vocab,
            evaluation=True,
            sort_during_eval=True
        )

        preds = []
        if len(batch) > 0:
            for i, b in enumerate(batch):
                preds += self.trainer.predict(b)
        preds = utils.unsort(preds, batch.data_orig_idx)
        batch.doc.set([HEAD, DEPREL], [y for x in preds for y in x])

        doc_conll = CoNLL.convert_dict(batch.doc.to_dict())
        conll_string = CoNLL.conll_as_string(doc_conll)
        return conll_string


def _hyperparam_search():
    args = DEFAULT_PARSER_ARGS.copy()

    def trial(args):
        train(args)
        las = test(args)
        return las

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
        vals = map(lambda x: str(x[0]), tt['misc']['vals'].values())
        las = str(1 - tt['result']['loss'])
        print('\t'.join([las, "\t".join(vals)]))
