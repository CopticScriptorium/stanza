# THIS IS A FORK!

This is [Coptic Scriptorium](https://github.com/CopticScriptorium)'s fork of Stanza. We use Stanza's dependency parser in a lightly modified form.

## Coptic Scriptorium-specific instructions

### Train a model
In this repository:
```
python coptic_cli.py train
# model will appear under `stanza_models/`
```
Or from your own code:
```python 
from stanza.coptic import train
train('/path/to/train.conllu', '/path/to/dev.conllu')
```

### Use a model
**NOTE**: You **must** make sure the parameters you use in production **exactly match** the parameters you used during 
training. Failing to do so will likely degrade performance. 

1. Download [`saved_models.tar.gz`](https://github.com/CopticScriptorium/stanza/tree/master/saved_models.tar.gz) 
and unpack it in your production working directory. 
2. `pip install --upgrade git+git://github.com/CopticScriptorium/stanza.git#egg=stanza`
3. Use the functions `train`, `test`, and `Predictor.predict` from `stanza.coptic`. These functions will accept 
**either** a conllu string or a filepath to a valid conllu file:
```python
>>> from stanza.coptic import train, test, Predictor
>>> # train a new model
>>> train('/path/to/train.conllu', '/path/to/dev.conllu')
>>> # eval on a dataset using saved model
>>> test('/path/to/test.conllu')
>>> # load saved model into memory and get predicted conllu strings
>>> p = Predictor()
>>> p.predict('my-conllu-string-or-conllu-filepath')
1	ⲁⲩⲱ	ⲁⲩⲱ	CONJ	CONJ	ForeignWord=No|MorphCount=1|Entity=O	10	cc	_	_
2	ϫⲉ	ϫⲉ	CONJ	CONJ	ForeignWord=No|MorphCount=1|Entity=O	3	mark	_	_
3	ⲛ	ⲡ	ART	ART	Definite=Def|Number=Plur|PronType=Art|ForeignWord=No|MorphCount=1|Entity=B-person	10	dislocated	_	_
4	ⲉⲧ	ⲉⲧⲉⲣⲉ	CREL	CREL	ForeignWord=No|MorphCount=1|Entity=I-person	5	mark	_	_
5	ⲛⲕⲟⲧⲕ	ⲛⲕⲟⲧⲕ	V	V	fin=fin|subord=subord|ForeignWord=No|MorphCount=1|Entity=L-person	3	acl	_	Orig=ⲛ︦ⲕⲟⲧ︤ⲕ︥
6	ⲅⲁⲣ	ⲅⲁⲣ	PTC	PTC	Position=Wack|ForeignWord=Yes|MorphCount=1|Entity=O	10	advmod	_	_
7	.	.	PUNCT	PUNCT	ForeignWord=No|MorphCount=1|Entity=O	5	punct	_	_
8	ⲉ	ⲉⲣⲉ	CCIRC	CCIRC	ForeignWord=No|MorphCount=1|Entity=O	10	mark	_	_
9	ⲩ	ⲛⲧⲟⲟⲩ	PPERS	PPERS	Definite=Def|Number=Plur|Person=3|PronType=Prs|ForeignWord=No|MorphCount=1|Entity=O	10	nsubj	_	_
10	ⲛⲕⲟⲧⲕ	ⲛⲕⲟⲧⲕ	V	V	fin=fin|subord=subord|ForeignWord=No|MorphCount=1|Entity=O	0	root	_	Orig=ⲛ︤̄ⲕ︥ⲟⲧ︤ⲕ︥
11	ⲛ	ⲙ	PREP	PREP	ForeignWord=No|MorphCount=1|Entity=O	13	case	_	Orig=ⲛ̄
12	ⲧⲉ	ⲡ	ART	ART	Definite=Def|Gender=Fem|Number=Sing|PronType=Art|ForeignWord=No|MorphCount=1|Entity=B-time	13	det	_	_
13	ⲩϣⲏ	ⲟⲩϣⲏ	N	N	ForeignWord=No|MorphCount=1|Entity=L-time	10	obl	_	Orig=ⲩϣⲏ̂
14	·	·	PUNCT	PUNCT	ForeignWord=No|MorphCount=1|Entity=O	10	punct	_	_
```

### Keeping this fork up to date

#### With Stanza
Use GitHub's [compare changes](https://github.com/CopticScriptorium/stanza/compare/master...stanfordnlp:master) interface
to generate a pull request from `stanfordnlp/stanza` to `copticscriptorium/stanza`.

Watch out for breaking changes--our custom code here was developed against `stanza==1.1.1`. Parser flags may change in
the future.

#### With Coptic Scriptorium's UD and lexical data
Update the files under `stanza/coptic_data`.

#### Training a new model
Train the model, make sure your new parameter settings are committed to `stanza/coptic.py`, and update `saved_models.tar.gz`.

<hr>

<div align="center"><img src="https://github.com/stanfordnlp/stanza/raw/dev/images/stanza-logo.png" height="100px"/></div>

<h2 align="center">Stanza: A Python NLP Library for Many Human Languages</h2>

<div align="center">
    <a href="https://travis-ci.com/stanfordnlp/stanza">
        <img alt="Travis Status" src="https://travis-ci.com/stanfordnlp/stanza.svg?token=RPNzRzNDQRoq2x3J2juj&branch=master">
    </a>
    <a href="https://pypi.org/project/stanza/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/stanza?color=blue">
    </a>
    <a href="https://anaconda.org/stanfordnlp/stanza">
        <img alt="Conda Versions" src="https://img.shields.io/conda/vn/stanfordnlp/stanza?color=blue&label=conda">
    </a>
    <a href="https://pypi.org/project/stanza/">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/stanza?colorB=blue">
    </a>
</div>

The Stanford NLP Group's official Python NLP library. It contains support for running various accurate natural language processing tools on 60+ languages and for accessing the Java Stanford CoreNLP software from Python. For detailed information please visit our [official website](https://stanfordnlp.github.io/stanza/).

🔥 &nbsp;A new collection of **biomedical** and **clinical** English model packages are now available, supporting syntactic analysis and named entity recognition (NER) from biomedical literature text and clinical notes. For more information, check out our [Biomedical models documentation page](https://stanfordnlp.github.io/stanza/biomed.html).

### References

If you use this library in your research, please kindly cite our [ACL2020 Stanza system demo paper](https://arxiv.org/abs/2003.07082):

```bibtex
@inproceedings{qi2020stanza,
    title={Stanza: A {Python} Natural Language Processing Toolkit for Many Human Languages},
    author={Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D.},
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    year={2020}
}
```

If you use our biomedical and clinical models, please also cite our [Stanza Biomedical Models description paper](https://arxiv.org/abs/2007.14640):

```bibtex
@article{zhang2020biomedical,
  title={Biomedical and Clinical English Model Packages in the Stanza Python NLP Library},
  author={Zhang, Yuhao and Zhang, Yuhui and Qi, Peng and Manning, Christopher D. and Langlotz, Curtis P.},
  journal={arXiv preprint arXiv:2007.14640},
  year={2020}
}
```

The PyTorch implementation of the neural pipeline in this repository is due to [Peng Qi](http://qipeng.me), [Yuhao Zhang](http://yuhao.im), and [Yuhui Zhang](https://cs.stanford.edu/~yuhuiz/), with help from [Jason Bolton](mailto:jebolton@stanford.edu) and [Tim Dozat](https://web.stanford.edu/~tdozat/).

If you use the CoreNLP software through Stanza, please cite the CoreNLP software package and the respective modules as described [here](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers) ("Citing Stanford CoreNLP in papers"). The CoreNLP client is mostly written by [Arun Chaganty](http://arun.chagantys.org/), and [Jason Bolton](mailto:jebolton@stanford.edu) spearheaded merging the two projects together.

## Issues and Usage Q&A

To ask questions, report issues or request features 🤔, please use the [GitHub Issue Tracker](https://github.com/stanfordnlp/stanza/issues). Before creating a new issue, please make sure to search for existing issues that may solve your problem, or visit the [Frequently Asked Questions (FAQ) page](https://stanfordnlp.github.io/stanza/faq.html) on our website.

## Contributing to Stanza

We welcome community contributions to Stanza in the form of bugfixes 🛠️ and enhancements 💡! If you want to contribute, please first read [our contribution guideline](CONTRIBUTING.md).

## Installation

### pip

Stanza supports Python 3.6 or later. We recommend that you install Stanza via [pip](https://pip.pypa.io/en/stable/installing/), the Python package manager. To install, simply run:
```bash
pip install stanza
```
This should also help resolve all of the dependencies of Stanza, for instance [PyTorch](https://pytorch.org/) 1.3.0 or above.

If you currently have a previous version of `stanza` installed, use:
```bash
pip install stanza -U
```

### Anaconda

To install Stanza via Anaconda, use the following conda command:

```bash
conda install -c stanfordnlp stanza
```

Note that for now installing Stanza via Anaconda does not work for Python 3.8. For Python 3.8 please use pip installation.

### From Source

Alternatively, you can also install from source of this git repository, which will give you more flexibility in developing on top of Stanza. For this option, run
```bash
git clone https://github.com/stanfordnlp/stanza.git
cd stanza
pip install -e .
```

## Running Stanza

### Getting Started with the neural pipeline

To run your first Stanza pipeline, simply following these steps in your Python interactive interpreter:

```python
>>> import stanza
>>> stanza.download('en')       # This downloads the English models for the neural pipeline
>>> nlp = stanza.Pipeline('en') # This sets up a default neural pipeline in English
>>> doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
>>> doc.sentences[0].print_dependencies()
```

The last command will print out the words in the first sentence in the input string (or [`Document`](https://stanfordnlp.github.io/stanza/data_objects.html#document), as it is represented in Stanza), as well as the indices for the word that governs it in the Universal Dependencies parse of that sentence (its "head"), along with the dependency relation between the words. The output should look like:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

See [our getting started guide](https://stanfordnlp.github.io/stanza/installation_usage.html#getting-started) for more details.

### Accessing Java Stanford CoreNLP software

Aside from the neural pipeline, this package also includes an official wrapper for acessing the Java Stanford CoreNLP software with Python code.

There are a few initial setup steps.

* Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and models for the language you wish to use
* Put the model jars in the distribution folder
* Tell the Python code where Stanford CoreNLP is located by setting the `CORENLP_HOME` environment variable (e.g., in *nix): `export CORENLP_HOME=/path/to/stanford-corenlp-4.1.0`

We provide [comprehensive examples](https://stanfordnlp.github.io/stanza/corenlp_client.html) in our documentation that show how one can use CoreNLP through Stanza and extract various annotations from it.

### Online Colab Notebooks

To get your started, we also provide interactive Jupyter notebooks in the `demo` folder. You can also open these notebooks and run them interactively on [Google Colab](https://colab.research.google.com). To view all available notebooks, follow these steps:

* Go to the [Google Colab website](https://colab.research.google.com)
* Navigate to `File` -> `Open notebook`, and choose `GitHub` in the pop-up menu
* Note that you do **not** need to give Colab access permission to your github account
* Type `stanfordnlp/stanza` in the search bar, and click enter

### Trained Models for the Neural Pipeline

We currently provide models for all of the [Universal Dependencies](https://universaldependencies.org/) treebanks v2.5, as well as NER models for a few widely-spoken languages. You can find instructions for downloading and using these models [here](https://stanfordnlp.github.io/stanza/models.html).

### Batching To Maximize Pipeline Speed

To maximize speed performance, it is essential to run the pipeline on batches of documents. Running a for loop on one sentence at a time will be very slow. The best approach at this time is to concatenate documents together, with each document separated by a blank line (i.e., two line breaks `\n\n`).  The tokenizer will recognize blank lines as sentence breaks. We are actively working on improving multi-document processing.

## Training your own neural pipelines

All neural modules in this library can be trained with your own data. The tokenizer, the multi-word token (MWT) expander, the POS/morphological features tagger, the lemmatizer and the dependency parser require [CoNLL-U](https://universaldependencies.org/format.html) formatted data, while the NER model requires the BIOES format. Currently, we do not support model training via the `Pipeline` interface. Therefore, to train your own models, you need to clone this git repository and run training from the source.

For detailed step-by-step guidance on how to train and evaluate your own models, please visit our [training documentation](https://stanfordnlp.github.io/stanza/training.html).

## LICENSE

Stanza is released under the Apache License, Version 2.0. See the [LICENSE](https://github.com/stanfordnlp/stanza/blob/master/LICENSE) file for more details.
