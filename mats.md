# TKO 7095 Intro to human language technology

Notes from lecture slides

## HLT

### What is HLT

Natural language equals human language, different from formal and artificial languages such as programming languages

2 key challenges

* Variety:
  * Same meaning can be expressed in many different ways
  * You don't get it - You don't understand it
* Ambiguity
  * Same form can express different meanings based on context
  * different meanings for eg. tie

### NLP

Computational methods to analyze, understand or generate human language

#### Applications

Language is the most efficient way we have to communicate _meaning_

* Reasonably easy to describe a picture or music using words
* Very hard to describe music using a picture, or a book using dance

A large portion of what is presented as AI is actually build on NLP models:

* DALL-E
* MusicLM

Text correction and generation

* Spelling and grammar check
* Autocorrect and predictive text
* ChatGPT

Web search includes a lot of NLP

* Disambiguation
* Text classification
* Information retrieval
* Question answering
* Information extraction

Machine Translation

* DeepL

Chatbots and generative AI

Text to speech

* Eg. speak written books with ai machine voice

Speech recognition and subtitling

Filterin and moderation support via:

* Sentiment analysis
* Text filters

#### Language as data

##### Methods

* Simple statistics == counting
  * How many words / chars
  * word frequencies

* Segmentation == Divide bigger units into smaller ones
  * Almost always part of raw text preprocessing in NLP pipeline
  * _Tokenization_ / _word segmentation_
    * Segment text into individual tokens == word-like elements
    * _Sentence splitting / sentence segmentation_ == Segment text into individual sentences
    * _Stemming / lemmatization_ == reduce words to base forms
    * _Syntactic analysis / parsing_ == Identify syntactic structure of sentences

## CORPORA

A corpus is a _structured_ and _documented_ collection of language data

* Text corpus == collection of written text
* Speech corpus == collection of spoken language (audio)

Corpora come in a broad variety of types, from minimally structured (doc boundaries only) to comprehensively analyzed (Universal dependencies)

Corpora are the fuel that powers NLP, and the ability to use and create corpora if necessary is a key skill

### Definitions

Text corpora can be broadly categorized by their annotation

* Raw text corpora: Plain, unstructured
* Annotated text corpora: markings from some information of intrest

Annotations can be almost anything: Language, domain, genre, topic, sentence/token boundaries, morphology, syntax, fluency, sentiment, emotion, toxicity, names, dates, addresses, relations, events.

Annotations can mark elements at various levels of granularity: document, paragraph, sentence, token, character

#### Corpora

Can be categorized into two different things:

* Manually annotated corpora: Annotations made by humans
* Automatically annotated corpora: Annotations made by a machine

Also it's common for manually annotated corpora to use some level of automation == Though correction of machine-predicted annotations.

Manually annotated corpora are generally preferred for _quality_, but typically have much higher _cost_ to create

#### Unannotated corpora

Raw text corpora without annotation are broadly used in NLP, not only as material for _linguistic analysis_ but also for _training language models_.

* Large neural models trained on unannotated text (eg. to predict next word) are the basis for many breakthrough advances in NLP over the last 5+ years
* For example GPT-3 was trained only on unannotated data (mostly web crawl), while ChatGPT and GPT-4 training also includes annotated data.
* The size of corpora that can be effectively used in NLP has increased massively in the last decades: BNC(British national corpus) 100milloin words at -94, now biggest models use more than trillion words.

#### Annotated corpora

In the central of developing NLP methods:

* _Evaluation_ to determine how well a method performs, its annotations are compared to human annotations ("gold standard")
* _Training_ for machine learning based approaches human annotations are used to train the ML methods.
* The type of annotation needed depends on the NLP task

#### Representation

Representations of annotations can be grouped into two primary categories:

1) Inline annotations: Inserted directly into the text HTML / XML
2) Standoff annotations: Stored separately from the tet; annotated parts of text are identified e.g. through character offsets

## Text annotation

### Annotation tasks

* Text classification: Genre, topic, sentiment ...
* Token/span classification: part-of-speec, named entity recognition, chunking
* Token/span normalization: entity linking, lemmatization, word sense disambiguation ...
* Relation annotation: Dependency syntax, entity relation, coreference
* Free-text annotation: Question answering, translation, summarization

Corpora frequently combine multiple categories of annotation: for example a corpora
annotated for information extraction may include named entity, entity relation and normalization annotation

#### Text classification

Assign each _text_ a label or a set of labels from predefined categories, e.g. genres

#### Token classification

Assign each _token_ a label or a set of labels from predefined categories. Data can be represented as sequence of (token, label) pairs potentialy organized into sentences == part-of-speech tagging

#### Span classification

Assign spans of tokens labels from predefined categories

Data can be represented as sequence (start,end,label) tuples or Begin-In-Out tags, for example NER

##### Span classificataion: BIO tags

Begin-In-Out tagging is frequently used to represent annotation that marks (non-overlapping) sequences of tokens

* Begin: Start of annotated span
* In: token inside annotated span
* Ou: not part of annotated span

Also reduced form IO (In-Out) and extended form IOBES (+End-Single)

#### Span normalization

Associate _spans of tokens_ with identifiers from external resources

Data can be represented as sequence of (start, end, id) tuples, for example enitity linking

#### Relation annotation

Associate tokens or spans with other tokens or spans

Data can be represented as sequence of (from, to, relation) tuples, it assumes identifiers from the from and to items

Dependency relations

#### Free-text annotation

Associate corpus text with arbitrary output text

Data can be represented as (input-text, output-text) pairs

Machine translation

### Annotation process

A small annotation project for a well-defined target may be completed by a single person in hours (e.g. spam marking)

Larger annotation projects frequently involve several annotators, tens of thousands of annotations or more. It can take months or years and requre careful planning

* Annotation coordinator or lead annotator oversees the process and makes final decisions in cases of disagreement
* Annotation guideline documents the task, overall guidelines, and specific decisions made to resolve difficult or diagreed cases
* Inter-annotation agreement is used to monitor annotation quality and consistency

#### Possible workflow

* Select annotation targets, representation and tools
* Select texts to annotate (may overlap with annotation)
* Draf annotation guidelines
* Hire annotators
* Initial annotator training, measuring inter-annotator agreement
* Refinement of annotator guidelines
* Primary annotation process including:
  * Monitoring inter-annotator agreement, identifying sources of disagreement
  * Updating guidelines, revisiting prior annotation if changed
  * Training new annotators
* Final consistency checking, documentation and release

#### Text sources

Most text corporas are build on pre-existing texts rather than creating new ones

* Allows capturing real language use, avoids effort and expense of creating text

In NLP, corpora used to be build on newswire, newspaper and other traditionally published sources, but recently increasingly on internet sources

The choise of text source depends on goals: for example, unannotated corpora tend to emphasize _size_, while annotated corpora emphasize _quality_

## Document representation and classification

### Classification

Process:

* IN: document
* OUT: label(s)

To be understood in a very broad sense

* Document can be anything from a few words to a whole book
* Label can be anything from positive/negative to hierarchies of tens of thousands of classes

Binary classification: One positive/negative label to be predicted, a special case of multi-class with class number == 2
Multi-class classification: one label from a larger label vocabulary to be predicted
Multi-label: a number of labels to be predicted from a larger label vocabulary

#### Binary classification

* Spam detection
* Sentiment analysis (pos/neg)
* Fake news detection
* Churn prediction
* Legal document classification
* Clickbait Detection

#### Multi-class classification

* Topic categorization
* Language identification
* Handwriting recognition
* Emotion detection
* Product actegorization
* Customer segmentation
* Document genre classification
* Essay grading
* ChatGPT :o'

#### Multi-label

* Set of topics to a document from a larger topic category
* MeSH Medical Subject Headings classify each biomedical publication with 10-15 terms
* News article tagging
* Medical condition prediction
* Customer support ticket tagging

### Document classification as a ML problem

* The text document needs to be reduced into a set of features
* A classifier can then be trained on the task as usual
* Input representation to a classifier tends to be fixed-length: a feature vector
* The space of features is enormous (one feature for each possible word in the language leading to a very sparce vector)
* Language tends not to be fixed

#### BoW Bag of Words representation

The feature vector has as many elements as there are words in the vocabulary. A non-zero value is set for words present in the document, zero for words absent. Any standard classifier can use this representation as its input

PLUS:

* Simple and fast to compute
* Works with any standard classifier
* Deceptively simple -> may give suprisingly good results in many keyword-driven classification tasks
* In many applications this can be the most bang for the buck solution!

MINUS:

* No encoding of order: "the dog chases the cat" has same representation as "the cat chases the dog"
* Long, sparse feature vectors are a challenge for any ML techniques

#### Multi-layer preceptron MLP

One input for each vocabulary item (possible word) is set to 0 if absent, non-zero if present

##### MLP in NLP

The input layer width is massive - one inpt for each vocabulary item. Most inputs are set to zero, only those seen in the input text are set to a non-zero value. It is more natural to think in terms of "embeddings". Each possible input is assigned a trainable vector, its embeddings. These can then be summed, producing the input representation. This is fully equivalent to the traditional definition of MLP, the embedding simply corresponds to the weights fanning out from a single input node in the "classical" MLP

##### Embeddings

Embeddings is a central concept in NLP.

A learned representation of an input feature, typically a dense vector of some hundreds to low thousands in length. Typically starts its life as a randomly initialized embedding matrix, then adjusted by the model during its training, and saved together with the model. The size of the embeddings matrix is one of the factors which dictate that we should keep our vocabulary quite limited (typically tens of thousands of items)

### Basic training pipeline - the practicalities

* Data with text and labels usually split in to train, validation and test sections
* Tokenize the text and map onto a vocabulary (tokens into integers)
* Batch examples and pad sequences of unequal length
* Design a model
* Train the model on the train data

#### Data

Typically list of dictionaries, keys for the various properties for each datapoint. In classification, minimally text and class label.

If not pre-divided into train-val-test sections, then you need to divide the data yourself

#### Tokenize and map vocabulary

* Split text in to words
* Map words into numerical indices in a vocabulary
* You may need to build the vocabulary too
* The class name also needs to be mapped into a 0-based integer

#### Batch and pad

* NLP is mostly based on neural networds and these are trained on batches of examples
  * Computational efficiency on GPU
  * Numerical stability of the gradient descent optimization
* Examples are represented as tensors (quite often just vectors and matrices)
* Text examples are not same length and must be padded to form a tensor
* Must make sure the padding "0" is understood throughout as a special value, and not a token index

#### Model

We will use torch to implement our models and the HuggingFace libraries so we can naturally continue in the Deep Learning course

The model is a class

* __init__(args) initializes the learnable parameters (layer weight matrices, embeddings etc)
* forward(inputs) carries out the computation on a single batch, computes loss
* Generally speaking, forward() will receive the contents of one batch dictionary at a time

#### Training

Runs the training loop

* Batch of data is fed into the model's forward function
* The function calculates the output for the batch and if labels are given also the loss
* The loss is almost uniquely crossentropy in the simple classification case
* Model weights are updated by a step in the opposite direction of the loss gradient, the length of the step depends on the learning rate witch is a parameter

The Trainer class automatized this for us.

## Sequence labeling

### SL Definitions

Basic task setting:

* Input: text represented as sequence of tokens
* Output: label(s) for each _token_ from predefined categories

For most tasks, exactly one label per token

Contrast with text/document classification: Label(s) for _text as whole_

#### Terminology

* Some sequence labeling tasks termed "token classification"
  * Note potential confusion with "sequence classification" for document classification
* Label, class and tag are here _largerly synonymous_ depending on task

### SL Tasks

* POS - Part Of Speech tagging, labels: VERB, NOUN, ADJ, ...
* NER - Named Entity Recognition, labels: PERSON, LOCATION, ORGANIZATION, ...
* Chunking / partial parsing, spans corresponding to for example verb and noun phrases

Also segmentation, morphological tagging, semantic role labeling, text zoning (introduction, methods, results, etc.) Generally any task involving marking _spans_

### SL as feature generation

In a "traditional" NLP pipeline, sequence labeling tasks used in part to create features for later tasks, for example:

* Parts of speech serve as features to indentify noun and verb phrases (chunking)
* Parts of speech and chunk tags serve as features to identify named mentions

Explicit features introduced in this way generally not used in recent deep learning approaches, here still used for illustration and interpretability

#### SL Challenges

Ambiguity and context dependence:

* POS tagging: "can" as noun vs. aux verb, "house" as noun vs. verb
* NER: "Washington" as person vs. location, "Nokia" as organization vs. location

Label dependencies:

* POS (DET, NOUN) likely e.g "the dog", (DET,VERB) unlikely e.g "this is"
* NER (B-PERSON, I-PERSON) is valid, (B-PERSON, I-ORGANIZATION) is invalid

### SL Representation

Build on ideas from document classification:

* Supervised machine learning: train model based on annotated corpus
* Explicitly defined features: manually build appropriate features for task

By contrast to document classification approach, we _cannot_ apply a simple bag-of-words approach to sequence labeling and
hope for good results

* Losing order of words would make many cases _undecidable_
* Consider "We can see a can"

We _cannot_ simply consider each token as its own "document" == needs context, or just make a BOW from a whole sentence == cannot differentiate!

Instead we'll here rely on two well-tested ideas for representing text for sequence labeling:

1) Context windows: Create features form fixed-size "window" of tokens before and after each token we're classifying
2) Relative positions: Represent the position of context tokens and their features with respect to the token we're classifying.

#### SL Context window

Most sequence labeling decisions are comparatively _local_, with distant tokens contributing little useful information

* Especially POS tagging and other morphology-level labeling largely sentence-internal

For classifying a token, only represent information from its immediate context instead of e.g entire document.

#### SL Relative positions

The _order_ in which tokens appear in the input is key to many sequence labeling tasks, POS tagging for example.

* "can see a..." -> "can" is auxiliary verb
* "see a can..." -> "can" is noun

The key piece of information is _not_ absolute position in the input, but relative position with respect to the token being labeled

Explicitly encode relative position to focus word in features

##### Explicit features

For document classification each word form was used as its own separate feature

* Features include eg ```cat```and ```dog```for words _cat_ and _dog_

More generally we can explicitly define any feature we like

We'll here represent these in text, for example:

* ```token[-1]=dog```the preceding token is "dog"
* ```pos[+1]=NOUN```the next token is a noun (POS tag has value NOUN)
* ```chunk[0]=B-NP```the current token starts a noun phrase

Note: these are _text strings_ representing the presence of a feature; there is nothing special about their form.

Explicit feature representations allow us to represent arbitrary categorical information about the focus token and context tokens, e.g.

* Does the token start with a capital letter?
* What are the last two characters of the token?
* What two-word sequence (bigram) starts at the token?
* Does the token appear in a list of known names?
* Does the token consist only of letters / digits / puncts?

Categorical features can also be introduced for the window, sentence or document as a whole, and for some ML methods features can also be given weights TF/IDF

### SL Sum

Each token in the dataset form its own example of classification, where:

* Features are generated from a _fixed-size window_ around the token
* _Relative position_ with respect to the focus token is represented to capture word order and distance
* _Explicit features_ are defined to represent relevant information about the focus token and other tokens in the window (surface form)

We can use these as a "bag of features" with the MLP classifier

#### What's still missing?

For MLP represented is:

* Context window (locality)
* Relative word order (sequence order, distance)
* Arbitrary categorical aspects of the input (explicit features)

However, predictions for each token were made _independently_ of each other

* No attempt to model label dependencies
* May predict very unlikely or invalid label sequences

### SL Methods

Sequence labeling tasks can be addressed through _heuristic_ and _rule-based_ approaches

* For example, dictionary-based approach to NER: compile lists of known names with types (person etc), find occurences in the text

State-of-the-art sequence labeling methods are based on supervised ML. Broadly speaking almost any ML method _can_ be used for sequence labeling. Methods that inherently capture sequential nature of data and can model _label dependencies_ are particularly good fit

Methods capturing sequential order and/or label dependency information include:

* Hidden Markov Models (HMM)
* Conditional Random Fields (CRF)
* Recurrent Neural Network (RNN)
  * Long Short-Term Memory (LSTM)
  * Gated Recurrent Unit (GRU)
* Transformer models
  * Bidirectional (BERT)
  * Encoder - Decoder (T5)

### SL Models

#### Basics of graphical models

![first](grap1.png)
Two states (Sunny and cloudy) with probabilities of transitioning between the two ased on the current state: a _first-order_ markov model

For each state, we have a probability of _starting_ the prcoess in the state

![second](grap2.png)

For each state, we have probability of _emitting_ a particular _observation_

![third](grap3.png)

#### HMM

First-order Hidden Markov MOdel (HMM) is defined by:

* Set of (hidden) states X_1,...,X_n and possibe outputs Y_1,.. Y_m
* Start probabilities for each state
* Transition probabilities between states
* Emission or output probabilities for each state and output

Given this information, we can answer questions such as:

* What is the probability of a particular state sequence?
* What is the probability of a particular output?
* What is the probability of a state sequence _given_ output?

In basic application to sequence labeling, the hidden states Y_i correspond to the labels (POS or NER tags for example) and the outputs X_i to tokens

Parameters can be straightforwardly set given annotated data

#### Metrics

_Token-level_ classification _accuracy_ (correct predictions out of all predictions) generally used to evaluate task

For tasks invlving marking _spans_ (NER) performance typically measured on span level in terms of exact-match precision, recall and F_1 score

* Compare predicted and gold standard spans in terms of (start-token, end-token, type)
  * Only triples where all values match between predicted and gold are correct
* Precision: fraction of predicted spans that are correcnt
* Recall: fraction of gold standard spans that are correctly predicted
* F_1-score: balanced harmonic mean of precision and recall

