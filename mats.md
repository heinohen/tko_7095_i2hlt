# TKO 7095 Intro to human language technology

Notes from lecture slides

## HLT

### What is HLT

Natural language equals human language, different from formal and artificial languages such as programming languages. Human language is natural or organic and does not follow a certain syntax in cotrast to a programming language. Only grammar to use as guidelines

2 key challenges

* _Variety_:
  * Same meaning can be expressed in many different ways. Same idea can be expressed in multiple different ways for example using different sentece structures or different words
  * Much obliged - Thank you
  * You're welcome - You got it - Anything for you
  * You don't get it - You don't understand it
  * Leave a message, i'll get back to you - Leave me a message and i will call you back
* _Ambiguity_
  * Same form can express different meanings based on context
  * WORD LEVEL AMBIGUITY:
    * different meanings for a single word eg. tie:
    * VERB: attach or fasten with string or similar cord
    * NOUN: a strip of material worn round the collar and tied in a knot
    * NOUN: a result in a game in which two or more competitors have the same score
  * SENTENCEL LEVEL AMBIGUITY:
    * Kids make nutritious snacks: Are the kids making the snacks or used as snacks?
    * She killed the man with the tie: Was the man wearing the tie or was it the murder weapon

### NLP

Computational methods to analyze, understand or generate human language

Computer science approach: Create methods for NLP tasks
Linguistic approach: Use the methods created to analyse some linguistic side problem

#### Applications

Language is the most efficient way we have to communicate _meaning_

* Reasonably easy to describe a picture or music using words
* Very hard to describe music using a picture, or a book using dance

A large portion of what is presented as AI is actually build on NLP models:

* DALL-E -- Creates pictures based on natural language prompt given
* MusicLM -- Creates music based on natural language prompt

Text correction and generation

* Spelling and grammar check
* Autocorrect and predictive text
* ChatGPT

Web search includes a lot of NLP

* Disambiguation -  process of distinguishing between similar things, in other words context of the search query
* Text classification - ""safe search"" allows to filter web pages with predictive labels
* Information retrieval - matching relevant web pages with the search query
* Question answering - google search directly generates answer with generative NLP task
* Information extraction

Machine Translation

* DeepL

Chatbots and generative AI

Text to speech from example a.i.mater

* Eg. speak written books with ai machine voice

Speech recognition and subtitling

Speech regocnition model listens audio and transcribes text. Used for youtube for example

Filtering and moderation support via:

* Sentiment analysis
* Text filters

#### Language as data

##### Methods

* Simple statistics == counting
  * How many words / chars
  * word frequencies

* Segmentation == Divide bigger units into smaller meaningful ones: eg. book in to chapters or sections, sections to paragraphs, paragraphs in to sentences, sentences in to words, words in to chars.
  * Almost always part of raw text preprocessing in NLP pipeline!
  * _Tokenization_ == word like element without punctuations / _word segmentation_
    * Segment text into individual tokens == word-like elements
    * _Sentence splitting / sentence segmentation_ == Segment text into individual sentences. Why: for example sentence parsing for analysing how it is built
    * _Stemming / lemmatization_ == reduce words to base forms
    * _Syntactic analysis / parsing_ == Identify syntactic structure of sentences

###### Tokenization

Tokenization workflow:

1) Split text from whitepsace characters, taken into account punctuation
2) Regular expressions: define search patterns, find these from raw text or find-and-replace if needed
3) Find all punctuation characters and replace with whitespace + punctuation character
4) Usually _it is not that important_ how exactly you do it, just be consistent!

* Works quite well for English, Finnish, Swedish, approx 97-99% on clean text
* Many tokenizers are just a large number (in the hundreds) of regular expressions

But does not work for all

All languages do not use whitespace or punctuation or the meaning of those may be different

* Chinese, thai, vietnamise

Naive algorithm 1

1) Build a vocabulary for the language
2) Start from the beginning of the text and find the longest matching word
3) Split the matching word and continue from the next remaining character

the table down there --> thetabledownthere --> theta bled own there

Does not work for English, but in Chinese words are usually 2-4 chars long so the simple algorithm works

Where to get the dictionary?

###### State-of-the-art SOTA

The best existing method currently known.

* Machine learning
  * Collect raw (untokenized) text from language you are in interested (news articles, internet) and manually tokenize it
  * Train a classifier
  * The trained classifier can be used to tokenize new text

###### Sentence splitting

Naive method 1: What kind of punctuation characters end the sentence?

* Yes: ?!.
* No: ,

Define a list of sentence-final punctuation, and always split on those. Problems?

Dot can be used in context such as abbreviations

Solution: Define a list of rules to indentify when punctuation does not end a sentence

* List of known abbreviations, list of regular expression to recongnize numbers
* How about missing puctuation ? Other languages ?

###### State-of-the-art

* Machine learning
  * Collect raw text fro the language you are interested in, and manually sentence segment it.
  * Train a classifier
  * The trained classifier can be used to sentence sengment new text

## CORPORA

A corpus is a _structured_ and _documented_ collection of language data

* Text corpus == collection of written text
* Speech corpus == collection of spoken language (audio)

Corpora come in a broad variety of types, from minimally structured (doc boundaries only) to comprehensively analyzed (Universal dependencies)

Corpora are the fuel that powers NLP, and the ability to use and create corpora if necessary is a key skill
This is because you are not able to use any machine learning based tools if you cannot train a model using a corpus! Or to evaluate performance of your model

### Definitions

Text corpora can be broadly categorized by their annotation

* Raw text corpora: Plain, unstructured
* Annotated text corpora: markings from some information of intrest to addition of the raw text

e.g.: collection of documents in different language with language name given as annotation.

Annotations can be almost anything: Language, domain, genre, topic, sentence/token boundaries, morphology, syntax, fluency, sentiment, emotion, toxicity, names, dates, addresses, relations, events.

Annotations can mark elements at various levels of granularity: document, paragraph, sentence, token, character

#### Corpora

Can be categorized into two different things:

* Manually annotated corpora: Annotations made by humans
* Automatically annotated corpora: Annotations made by a machine

Also it's common for manually annotated corpora to use some level of automation == Through correction of machine-predicted annotations.

Manually annotated corpora are generally preferred for _quality_, but typically have much higher _cost_ to create

#### Unannotated corpora

Raw text corpora without annotation are broadly used in NLP, not only as material for _linguistic analysis_ but also for _training language models_.

* Large neural models trained on unannotated text (eg. to predict next word) are the basis for many breakthrough advances in NLP over the last 5+ years
* For example GPT-3 was trained only on unannotated data (mostly web crawl), while ChatGPT and GPT-4 training also includes annotated data. The annotated data is used to fine-tune the models.
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
* Token/span classification: part-of-speech, named entity recognition, chunking
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

* IN: document - Document can be anything from a few words to a whole book
* OUT: label(s) - can be anything from positive/negative to hierarchies of tens of thousands of classes

To be understood in a very broad sense anything between a sentence and a book

Multi-label: a number of labels to be predicted from a larger label vocabulary

#### Binary classification, classes == 2, one predicted

Only two possible outcomes 0 / 1. Either document has this property or not e.g. negative or positive, spam or not spam. A special case of multi-class with class number == 2. Usually a simplification of the real problem, sentiment is not black and white, fakenews is not usually quite so binary.

* Spam detection
* Sentiment analysis (pos/neg) - tweets for example
* Fake news detection - fake or not
* Churn prediction - a measurement of the percentage of accounts that cancel or choose not to renew their subscriptions
* Legal document classification - is document relevant to my case or not
* Clickbait Detection - is this newspiece that google wants to show in its newsfeed this is actual news or a clickbait for advertisement of sorts

#### Multi-class classification, classes > 2, one predicted

ONE SINGLE LABEL from a larger label base to be predicted

* Topic categorization - piece of news -> assign exactly one topic
* Language identification - large web crawl of data, label documents with language indentification
* Handwriting recognition - image identification, identifying letters from handwriting alphabets A,B,C etc. as _labels_
* Emotion detection - spectrum of emotions 8 - 10 and you want to decide what emotion this document representes
* Product categorization
* Customer segmentation
* Document genre classification - news, advertimement e.g. for labels
* Essay grading - predict a grade between 1-5
* ChatGPT :) - multi class classification generating text. predicts the next word as a classification problem 80000 classes

#### Multi-label classification, classes > 2, MULTIPLE classes

MULTIPLE labels predicted from a larger base of labels

* Set of topics to a document from a larger topic category
* MeSH Medical Subject Headings classify each biomedical publication with 10-15 terms
* News article tagging
* Medical condition prediction
* Customer support ticket tagging

### Document classification as a ML problem

* The text document needs to be reduced into a set of features

Then _input_ for the ML classifier is a set of features and _output_ is some decision. The classifier can be trained based on this. One fundamential problem exists with ML models excluding the most modern ones: ML model expects a fixed length input === a feature vector.

* Input representation to a classifier tends to be fixed-length: a feature vector. The space of features is enormous, one feature for each possible word in the language leading to very sparce feature vectors and that is a problem because the language tends not to be fixed.

After creating the feature vector for whole _language_, and not the text then you can go through your text and see if it has this or that feature.

#### BoW Bag of Words representation

The feature vector has as many elements as there are words in the vocabulary meaning the whole languages vocabulary. One element for each word and a non-zero value is set for every word that actually exists in the document.

Different approaches:

* Set a non-zero (e.g. 1) if the word exists how many ever times
* Set a count for each word occurence

Any standard classifier: decision tree, svm, perceptron can eat a representation like this because it is fixed!

PLUS:

* Simple and fast to compute
* Works with any standard classifier
* Deceptively simple -> may give suprisingly good results in many keyword-driven classification tasks. A long document, not many classes.
* In many applications this can be the most bang for the buck solution! SPAM filtering e.g.

MINUS:

* No encoding of order: "the dog chases the cat" has same representation as "the cat chases the dog" (but many applications do not care about the order so much!)
* Long, sparse feature vectors are a challenge for any ML techniques

#### Multi-layer preceptron MLP

The simplest form of neural network. One input for each vocabulary item (possible word) set to 0 if absent in the document we are classifying, non-zero if present. One output for each possible class.

##### MLP in NLP

* Training happens between the layers in the learnable weights. After hidden layers the output is a probability distribution that sum up to one and the highest probability wins!
* Inputs -> weights -> net input function -> activation function -> output (non-linearity) -> as input for next layer. Hundreds hidden layers percent in modern systems
* The input layer width is massive - on input for each vocabulary item. Most inputs are set to zero, only those seen in the input text are set to a non-zero value.
* It is more natural to think in terms of "embeddings"
* Each possible input is assigned to a trainable vector, its embedding
* These can then be summed, producing the input representation
* This is fullu equivalent to the traditional definition of MLP, the embedding simply corresponds to the weights fanning out from a single input node in the "classical" MLP

###### MLP through the NLP lens

Input: "the dog chaces the cat" -> embedding matrix (hold all words in vocabulary of language) -> determine weights for each word in the document (here sentence) ->sum up the embeddings of seen words to create one "hidden layer" value (prior to non-linear transformation) === TRAINABLE!

##### Embeddings

What are these embedding vectors, can be clustered. Words that somehow are related to eachother seem to get similar embeddings === close together in vector space. The similarity of two embeddings in the vector space "cosinesim" "eucledian dist" seems to correlate to our understanding how two words goes to eachother with their meanings.
E.g the embedding for "dog" will be closer to embedding of "cat", than for example the embedding of "river". So embeddings have real meanings!

Embedding starts its life as a randomly initialized embedding matrix, then adjusted by the model during its training, and saved together with the model! The size of the embeddings matrix is one of the factors which dictate that we should keep our vocabulary quite limited (typically tens of thousands of items)

cross-entropy loss is the calculation to be used with classification

No word order in the embedding vectors!

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

* Input: text represented as sequence of tokens (most likely words)
* Output of the model is: label or multiple labels for _EACH_ token from predefined categories

For most tasks, exactly one label per token

Contrast with text/document classification: Label(s) for _text as whole_

#### Terminology

* Some sequence labeling tasks termed "token classification"
  * Note potential confusion with "sequence classification" for document classification
* Label, class and tag are here _largerly synonymous_ depending on task

### Sequence Labeling Tasks

* POS - Part Of Speech tagging, labels: VERB, NOUN, ADJ, ... most classical example of sequence labeling task.
* NER - Named Entity Recognition, labels: PERSON, LOCATION, ORGANIZATION, ...
* Chunking / partial parsing, spans corresponding to for example identifying what is a noun phrase or a verb phrase. Relevant text pre-processing task
* segmentation text in to pieces -- tokenization is done by ML methods that boil down to  _SEQUENCE LABELING_ instead of regular expressions.
* morphological tagging -- Morphological tagging is the task of assigning labels to a sequence of tokens that describe them morphologically. As compared to Part-of-speech tagging, morphological tagging also considers morphological features, such as case, gender or the tense of verbs.
* semantic role labeling -- who does what to whom in the sentence
* text zoning (introduction, methods, results, etc.)

Generally any task involving marking _spans_

#### Example Part of speech tagging

Identify _token spans_ constituting _mentions of names_ and assign the types:

* Often extended to include also mentions of e.g. as times and dates
* Note: names frequently span _multiple tokens_ in contrast of POS tagging!

Span start and extend typically marked using IOB (aka BIO) tags or variation such as IOBES (adds [E]nd , [S]ingle)

| item in span  | tag  |
|---|---|
| Barack  | B(egins)-Person  |
| Obama  |  I(nside)-Person |
| was  | O (outside)  |
| born  | O (outside)  |
| in  |  O (outside) |
| Hawaii | B-Location |

Super important subject BIO - IOB tagging!

Still used in fanciest models ever span encoding in text.

Also good to notice that certain sequences of tags are not legal (eg. B-Person followed by I-Location). Something to keep eye on when training the model.

IOB tagging can be applied to mark any _continuous, non-overlapping_ spans of tokens and assign them to categories:

* Phrases (chunks)
* Argumentative zones
* Semantic roles
* Hedged claims (e.g. "may...")

All of above can be generalized to NLP task TEXT ZONING.

Unable to tag embedded tags inside entites (University of Turku) is an entity but Turku is also location. Thats why many datasets use "longest span" method to limit the problem and just stick to the longest spans available and annotate that.

#### Character sequences

Sequence labeling in NLP not limited to _token_ sequences

Example, joint tokenization and sentence segmentation with labels

* _token-break_: token ends after character
* _sentence-break_: sentence ends after character
* _inside_: no break after character

INPUT: "Is it you?"

| char | tag |
|---|---|
|I|inside|
|s|token-break|
| |token-break|
|i|inside|
|t|token-break|
| |token-break|
|y|inside|
|o|inside|
|u|token-break|
|?|sentence-break|

### SL as feature generation

In a "traditional" NLP pipeline:

Sequence labeling is used as a pre-processing task to _generate features_ for the modules down the pipeline for e.g.:

* Parts of speech serve as features to indentify noun and verb phrases (chunking)
* Parts of speech and chunk tags serve as features to identify named mentions

Explicit features introduced in this way generally not used in recent deep learning approaches, here still used for illustration and interpretability

it used to be so that it is used to ENRICH THE INPUT FOR THE ML MODEL for later stages.

Nowdays, sequence labeling is happening implicitly inside large neural networks. nowdays you dony see this pipeline very often, still very useful for illustration and interpretability causes.

----> being phased out

#### SL Challenges

Ambiguity and context dependence:

* POS tagging:
  * "can" as noun SUBSTANTIIVI (container with soup) vs. aux verb APUVERBI (can happen)
  * "house" as noun (talo / koti) vs. verb (synonym for contain)
* NER:
  * PERSON: (george) "Washington" as PERSON vs. city or state in USA LOCATION
  * "Nokia" as ORGANIZATION vs. city in finland LOCATION

The actual beef of this task is to somehow take this _CONTEXT_ in to account!

Label dependencies:

* POS:
  * Determiner ("the", "a") + noun ("dog", "cat") pair (TARKENNE, SUBSTANTIIVI) (DET, NOUN) is very likely e.g "the dog". BUT Determiner + verb (DET,VERB) is very unlikely e.g "this is", "a is"

Dependencies across the labels are highly relevant!

* NER:
  * (B-PERSON, I-PERSON) is valid and quite likely sequence of labels, (B-PERSON, I-ORGANIZATION) is invalid, very unlikely and should practically never happen. And if it happens, it means that there is mistake in the data!

As for dataset, there usually is some of these errors inside.

But for model side it should "never" see sequences like that or predict sequence labels like that.

### SL Representation

Build on idea of the ideas from document classification:

* Supervised machine learning: Train model based on annotated corpus
* Explicitly defined features: Manually build appropriate features for task

In contrast to document classification it will be difficult to implement simple bag-of-words approach because there will not be enough of discriminating power for example

* "we can see a can" and you need to decide the POS tag for the word "can" in these two positions of the sentence. Thats why you just cannot feed this sentence as a BoW to a classifier because both of these _can_'s will have the same representation and you cannot make the distinction between.

LOSING ORDER OF WORDS WOULD MAKE MANY CASES

POS tagging is easy for 80% of the cases, but the remaining 20% is very hard

#### Problems

Representations as just one bag-of-words "document" per token for POS tagging would'nt work, because "can"(purkki) NOUN and "can"(voida) AUX would get the same (wrong) TAG

Representing the entire document as a bag-of-words wouln't work either because the most common tag i.e. PUNCTS
would get predicted for everything that would make the classifier very bad

We _CANNOT_ simply consider each token as its own "document" because we need context, OR just make one bag-of-words from the whole document because then we cannot differentiate.

Instead we'll here rely on two well-tested ideas for representing text for sequence labeling:

 1) Context windows: Create features from fixed-size "window" of tokens before and after each token we're classifying. In many tasks the desicion is relatively LOCAL. So whether a word is a NOUN or a VERB you usually don't have to look for 2-3 pages in both directions.
 2) Relative positions: Represent the position of _context tokens_ and their features with respect to the token we're classifying. Represent the positions of the things with respect to the word we are classifying

#### SL Context window

Most sequence labeling decisions are comparatively _local_, with distant tokens contributing little useful information.

This is good to use in especially POS tagging and other morphology-level labeling largely _SENTENCE-INTERNAL_

for example a sentence "We can house them in this house ."

if the window is on the first house. Look for one word behind and one word ahead: "can house them", in this context it is clear that the POS tag for the word under classifying is VERB (put something inside of smth).

Adn when the context window slides ahead for the end of sentence "this house ." We can see that the word house now gets a POS tag of a NOUN (a place to live in)

#### SL Relative positions

The _order_ in which tokens appear in the input is key to many sequence labeling tasks, POS tagging for example.

* "can see a..." -> "can" is auxiliary verb
* "see a can..." -> "can" is noun

``` python
-2   -1 0     +1    +2  +3    +4
We  can house them  in  this  house
```

The key piece of information is _not_ absolute position in the input, but relative position with respect to the token being labeled. It is HIGHLY relevant to know the word that is JUST before the word we are tagging or further away from it.

for example:

* CAN HOUSE: certainly we are looking at a AUX, VERB pair
* THIS HOUSE: certainly we are looking at a DET, NOUN pair

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

* Set of (hidden) states $X_1,...,X_n$ and possibe outputs $Y_1,...,Y_m$
* Start probabilities for each state
* Transition probabilities between states
* Emission or output probabilities for each state and output

Given this information, we can answer questions such as:

* What is the probability of a particular state sequence?
* What is the probability of a particular output?
* What is the probability of a state sequence _given_ output?

In basic application to sequence labeling, the hidden states $Y_i$ correspond to the labels (POS or NER tags for example) and the outputs $X_i$ to tokens

Parameters can be straightforwardly set given annotated data

#### Metrics

_Token-level_ classification _accuracy_ (correct predictions out of all predictions) generally used to evaluate task

For tasks invlving marking _spans_ (NER) performance typically measured on span level in terms of exact-match precision, recall and F_1 score

* Compare predicted and gold standard spans in terms of (start-token, end-token, type)
  * Only triples where all values match between predicted and gold are correct
* Precision: fraction of predicted spans that are correcnt
* Recall: fraction of gold standard spans that are correctly predicted
* $F_1$-score: balanced harmonic mean of precision and recall

## Language models

### LM Definitions

Task setting:

Learn to assign probabilities to sequences of words

* Input: corpus of raw, unannotated text (typically very large)
* Output model that can estimate $P(w_1, w_2, w_3, ... , w_n)$ i.e how likely is the word sequence in the language

Language modelling methods and uses changed dramatically over last decade

* Traditionally: n-gram models (bigram, trigram, ...) and approaches based on _counting_
* Recently: Neural network models (MLP, RNN, transformer) trained in _predicition_

Probability of word sequence estimated via probabilities of individual words in their context. Language models can be grouped into two broad categories by how that context is defined:

* Causal LMs: Estimate probability of word given previous words only only! (one-directiona, "left-to-right")
* Bidirectional LMs: Estimate probability of word given both preceding and following words

Broadly speaking, causal LMs are particularly effective in _generation_ and bidirectional LMs in _classification_

#### LM Tasks

##### Traditional tasks

* Spell and grammar checking: Detecting and correcting errors in text
* Short text generation: autocomplete, predictive text input
* Speech recognition: phoneme and word prediction
* Machine translation: ranking alternative translations
* Language identification: which language is text most likely in
* Text quality filtering

Broadly tasks involving _identifying unlikely spans_ of text or _ranking alternatives_ by their likelihood or belonging to a language

##### Modern tasks

* Chatbots: natular language dialogue on arbitrary topics
* Question answering: providing relevant responses to queries
* Summarization: short abstractive versions of input text
* Code completiong: software develpment support
* Machine translation: end-to-end translation, e.g. Finnish in, english out
* Zero- or few-shot tasks: e.g. text classification with very few examples

And many many more, increasingly any "low-lever" cognitive task that people can perform without training

## Count-based language models

Count-based modelling is a simple statistical approach to creating LMs. In the most typical case

* Input: sequence of words $(w_1 w_2 w_3 ...)$ representing large text corpus
* Output: model that can estimate probability of word given previous words $P(w_n | w_1 w_2 w_{n-1})$

Approach is estimating probability P using occurence counts C in corpus. There is an obvious issue with naively applying the probability estimate. Namely for every longer sequences of words, _the counts will be zero_. Key challenge for count-based LMs is the _sparsity of the data_, almost all interesting non-trivial texts will be entirely novel == new and original.

Example: What is the next word in the headline:
"Trump faces 'Real Danger Zone' from Mike Pence ..."?

### N-grams

_Data is finite_ and the counts of almost all longer word sequences will be zero.

Solution: instead of using the full "history" of previous words, only consider the previous _N_ words

* Bigram model: $P(w_n | w_{1:n-1})$
* Trigram model: $P(w_n | w_{n-2:w-1})$
* 4-gram model: $P(w_n | w_{n-3} w_{n-2} w_{n-1})$

Roughly in practice, bigram and trigram models simple to make but weak, 4- and 5-gram require substantial data, and larger than 7-gram rare due data requirements

N-gram model can be estimated from counts using _maximum likelihood estimate_

$P(w_n | w_{n-N+1:n-1}) = C(w_{n-N+1:n}) / C(w_{n-N+1:n-1})$

For example, for a bigram model we have simply:

$P(w_n | w_{n-1}) = C(w_{n-1:n})/C(w_{n-1})$

The probability estimate for a longer text can then be calculated using the _chain rule_

$P(w_1,...w_n) = \prod_{k \in \{1,...,n\}}P(w_k|w_{k-N+1:k-1})$

#### Smoothing and backoff

N-grams make count-based approach _possible_, but _zero counts_ will remain. Zero counts lead to $P(words) = 0$ estimates, which is useless for most applications.

Solution: apply _smoothing_ to either counts or probability estimates to avoid zeros

* Add-one smoothing == Laplace smoothing: simply add 1 to all counts
* Add-k smoothing: add fixed value k to counts (typically $k < 1$)
* Advanced smoothing methods: Knerser-Ney, Good-Turing, etc.

Alternatively when encountering a zero count, _back off_ from using an N-gram estimate to using an (N-1)-gram estimate

#### Numerical precision

Applying the chain rule to estimate the probabilities of a longer text involves _multiplying togeher many small probabilities_

Mathematically this is fine, but computers have limited ability to represent very small or very large values

remember float restrictions!

Solution: use log probabilities:

$p_1 \cdot p_2\cdot...\cdot p_n = exp(log(p_1)+log(p_2)+...+log(p_n))$

### Generation

Given a causal language model, it's straightforward to generate text:

1) Initialize _text_ to desired "prompt" or a start-of-text token (e.g. `<s>`)
2) Select next word _w_ from $P(w | text)$ and append it to _text_
3) Repeat previous step as long as desired (e.g. max-tokens or `</s>` sampled)

A naive selection strategy for step 2. is to always take the _most likely next word_ w:

$argmax_wP(w|text)$

BUT this type of _greedy decoding_ often gets trapped in repetition loops:

```"I don't know. I don't know. I don't know".```

and only produces "predictable" text. Strategies for better generation than greedy decoding include

* Randomly sampling the next word from the distribution $P(w | text)$
  * _Temperature_ parameter to (de)emphasize likely words
  * Limiting to _top-k words_ or a probability treshold (_top-p_ aka _nucleus sampling_)
* _Beam search_ to find likely sequences that start with a less likely word
* Filtering for redundancies, "bad words", etc

![generation](generation.png)

### Evaluation

Two common settings for LM evaluation:

1) Use _text to evaluate model_ or compare models, which assumes good text
2) Use _model to evaluate texts_, which assumes good model

For either case, the most common evaluation metric for LMs is _perplexity_ (PPL), the inverse probability of text normalized by the number of words

$PPL(w_1w_2...w_n) = P(w_1w_2...w_n)^{-1/N}=\sqrt[n]{\frac{1}{P(w_1w_2...w_N)}}$

PPL can be thought of as the weighted average number of words that can follow a word.

### N-gram limitations

N-gram models have been a key tool in NLP for decades, but have their clear limitations:

* _Limited use of context_ due to short history (N) and unidirectionality
* _Data sparcity_ means difficult estimating rare N-grams
* _High resource usage_ due to storage of large N-gram tables
* _No word similarity_ `cat != dog`, `cat != hat`
* Fixed vocabulary, out-of-vocabulary (OOV) words

## Prediction-based language models

Basic setup dentical to count-based models:

* Input: sequence of words $(w_1,w_2,...)$ representing large text corpus
* Output: model that can estimate probability of word given previous words $P(w_n | w_1,w_2,...w_n)$

Instead of deriving $P(words)$ from counts, predict it directly. Basically any method capable of learning to predict a probability distribution is applicable, but in practice focus on _neural network_ models: MLP, RNN, etc.

### LMs as representation learners

Language model solves in the end quite complex task. Prediciting the next word is by no means easy. Requires a good understanding of the language structure. Requires word knowledge (!) These properties make language modelling a good general task to learn representations of text units == embeddings

* Focus on _word_ embeddings
* In later courses we expand to longer text segments

#### Two-sided context

Language models are traditionally causal (left-to-right). This is due to how they were typically used in applications.

* E.g. speec recognition lattice decoder

For representation/embedding learning, we can also consider language models with two-sided context. This is due we do not benefit from causality, almost the contrary.

#### The word2vec model

`Yle website targeted in ____ attack`

Given this context, we want to predict the missing word, as a distribution. Word2vec is a very influential model in NLP. Makes two simplifying assumptions:

* Limit the left and right context window length
* Disregard word order

##### Language modelling as BoW classification problem

When you think about it, this is casting language modelling as a simple bag of words classification problem:

* Bag of Words as the input features
* Multiclass classification with every word in the vocabulary being one possible class
* Because we want the embeddings of input features / words

#### Word2vec continue

* A linear MLP
* Input/output layers as wide as the vocabulary
* Input-to-hidden layer weights learn embeddings
* Hidden-to-output layer weights discarded after training
* Embeddings are relatively short vectors
* $D(embedding length)$ is some 200-300

##### Word2vec training "Context BoW"

Can be trained as a normal classifier. Context goes in as bag of words. Word in the example is predicted.

##### Word2vec training "skip-gram"

Reduced to word pairs, given a single word, predict the distribution of words nearby. Thanks to the linear structure of the model, this is really not much different from CBOW above

#### W2V embeddings

Word2vec is trained on very large textual corpora. This is easy and very fast, the model is simple and includes several technical optimizations. After training, we retrain the embedding matrix of the model. Every vocabulary word thus has a single static learned vector embedding. "Similar" words get numerically similar embeddings.

##### W2V usage

* Primary use: pretrained representations that can be used to initialzie the embedding matrix of models for downstream tasks
* The embeddings encode information about the meaning of words, even if they are rare / missing in the supervised training data, which is often quite small

For example the word "crummy" might be seen only once in the IMDB training data, bot giving enouch opportunity to learn it as a negative feature. The word2vec embeddings are trained on much larger textual corpus, and the _embeddings for "crummy" is close to other negative words which the classifier may benefit from_.

The embeddings are affected by the w2v training data. Embeddings are one by word, conflating (== uniting) the different meanings of a ambiguous words. The embeddings are also _static_, unaffected by the present context in which the word is used.

##### Properties of embeddings

* Small context windows (+/-2): nearest words are syntactically similar words in same taxonomy.
  * Nearest neighbours for "hogwards" are other fictional schools: "sunnydale", "evernight", "blandings"
* Larger context windows (+/-5): nearest words are related words in same _semantic_ field:
  * Hogwarts nearest neightbors are Harry Potter world: Dumbledore, half-blood, Malfoy

##### Word analogy

Does not work nearly as often as you'd like, but it is very impressive when it does work.

Embeddings reflect the data meaning compute a bias for every adjective for example how much closer the adjective is to "woman" synonyms than "man" synonyms or names of particular ethnicities

* Embeddings for competence adjective (smart, wise, brilliant, resourceful, thoughtful, logical) are biased towards men, a bias slowly degreasing 1960-1990
* Embeddings for dehumanizing adjectives (barbaric, monstrous, bizarre) were biased towards Asians in the 1930s, bias decreasing over the 20th century

##### Alignment of embedding spaces

* Cross-lingual embeddings: a shared embedding space across languages
* The goal is to embed meaning, not words
* Same meaning (regardless of language) == same/near vector representation

Embeddings can shift in vector on timeline!

There are two general ways of alignment of embedding spaces.

1) Build an aligned space from scratch
2) Build two separate spaces and map them onto each other after the fact

##### Alignment though pseudowords

* Gather a multilingual corpus of texts
* Train a single set of embeddings on the multilingual data
* But there is no signal forcing alignments to emerge, the languages will end up living in their own subspaces
* Solution: Replace words by multilingual pseudowords
  * A large dog#koira runs in the yard#piha
  * Suuri dog#koira juoksee pihalla
* Then learn the embeddings the usual way, on the multilingual data
* The pseudoword embediings are forced to be the same across the languages
* And that has effect on other embeddings too, effectively "forcing" the spaces to align

##### Alignment through mapping

* Given two embedding spaces and a set of word/vector pairs between these, we can induce a tranformation matrix mapping from one space to another

##### Cross-lingual alignment

* in other words we need a set of word pairs between two languages, and their respective monolingual embeddings
* This can be seen as "training" data for inducing the mapping matrix $M$. The matrix $M$ minimizes the mean square error (MSE) between $SM$ and $T$ target
* Dimenionality does not need to match

#### Embeddings evaluation

There are many methods to induce word embeddings from text and different parameters for one method.

How can we evaluate the embeddings?

* Intrinsic: Direct evaluation of the properties of the embeddings
* Extrisic: Evaluation of the embeddings in a downstream task for example incorporating the embeddings in a classifier / tagger and observing the performance on some classification / tagging taks

##### Intrisic

* Word similarity
* Word analogy: `A is to B as C is to ___`
  * Implemented arithmetically as $B-A+C$
* Concet categorization
  * sandwich, tea, pasta, water should form 2 groups (food, drink)
* Outlier detection
  * breakfast, _cereal_, dinner, lunch

##### W2V influence to NLP

W2V and other similar models were an important stepping stop towards todays NLP. In particular the present-day GPT models can be seen as a culmination of work which set out to:

* Make embeddings not static == make them context-dependent
* Deal with out-of-vocabulary items

Very roughly the progression was Word2vec -> convolutional NNs -> recurrent NNs -> attention models / Transformer NN

## Deep learning

Deep learning is a hot topic, not only within ML and NLP.

NLP models based on deep learning are _in the news every day_,  althouh they're not always called NLP models

Deep learning is a subset of ML using neural networks with _multpile layers_

* The "deep" in deep learning just refers to the depth of the NN in terms of layers. It has nothing to do with "deep understanding"

Deep architechtures enable NNs to learn _hierarchical representations_, where successive layers capture progressively more complex patterns. The potential _benefits of deep NN architectures_ have been known about for long, but only started to be broadly realized over the last 10 years.

Learned _hierarchical representations are easy to show for image recognititon:

Input layer is pixels, other layers capture increasingly _high-level_ features. Classification (e.g. face recognition) is easy given high-level features. The layer-to-layer mappings are all learned (representation learning)

Hierarchical representation learning is seen also in deep NNs for language:

* Lower layers capture _POS and phrase structure information_
* Middle layers capture _entities and semantic roles_
* Upper layers capture _coreference and entity relations_

Deep learning architectures have been pursued for decades; major breakthroughs in the last decade facilitated by:

* _Big data_, in particular datasets derived from internet crawls
* Increase in _computational power_ especially GPU-accelerated training
* _Algorithmic improvements_, both in NN architectures as well as in activation function, optimizers, etc
* _Transfer learning_ especially _pretraining_ on unlabeled data followed by _fine-tuning_ on task-specific data

### Big data

"Traditional" machine learning methods are effective when trained on _comparatively small_ datasets but often fail to take advantage of massively larger data. On the contrary _Large NNs_ are mostly useless when trained on small datasets, but perform well on large data and _continue to improve_ with more data as long as network size is scaled with data.
s
As _Internet-scale datasets_ have become more common readily available and approaches _training on unannotated data_ more common, deep NNs are increasingly favored.

The size of training data continues to increase `["GPT-1": 4Gb, "GPT-2": 40Gb, "GPT-3": 400Gb, "GPT-4":"unk"]`

Largest training datasets have been on an exponential growth curve, with benefits from scale continuing to this day. The amount of _high-quality text_ likely to increasingly to become a _key limitation_ for the largest models, especially for smaller languages.

### Computational power

Large NN models are now almost excusively trained using _hardware acceleration_ (GPUs / TPUs)

* GPU == Graphics Processing Unit
* TPU == Tensor Processing Unit

GPUs (and TPUs) allow _massively parallel_ computation and NN architectures are increasingly designed specifically to take advantage of this.

Training the largest models now requires _supercomputers_. For example LUMI: 2560 GPU nodes with 4 AMD MI250X devices each (20480 "GPUs"). _Large GPU clusters_ and millions of euros of compute investment required to train largest language models. _Costs continue to increase_ as amount of compute used grows faster than its price drops.

### Model architecture

Prior to 2017 deep learning approaches to NLP mostly build on _recurrent neural networks_ (RNNs) with some applications of convolutional neural nets etc.

Issues with RNN architecture include:

* _Long-term dependencies_: Basic ("vanilla") RNNs are famously "forgetful", failing to make use of information from distant earlier tokens. RNN variants such as LSTMs and GRUs alleviate but do not solve the problem.
* _Sequential processing_: RNNs require computation to be completed for token $N$ before computation for token $N+1$ can start, which limits the ability to make use of the massive parallelism that GPUs allow.

In the 2017 the Transformer was proposed:

* _Attention_ allows very long contexts
* _Model scaling_ add layers / increase dimensinality
* Efficient _parallelism_ for large-scale GPU training
* _State-of-the-art (SOTA) results_, not just in NLP

The original transformer architecture is the direct foundation for _three major classes_ of NN models, _hundreds of architecture variations_ and _100K+ specific models_

### Transfer learning

Generally _knowing one task can make it easier to learn a related task_. In transfer learning, _information from one task is used when learning another_. For NNs generally by reusing weights. For example we can use word2vec vectors to intialize the embedding layer weights.

Key idea allowing very large NN models to be applied in NLP: _transfer learning with unnannotated data_

* Pre-training: Learn "pure" LM on very large corpus of raw text
* Fine-tuning: Train task-specific model on small corpus of annotated data

Contrast, _raw text_ available from web crawls for example on scale of trillion words $10^{12}$. For _manually annotated corpora_ a million words $10^6$ is already quite large

### Neural language models

Deep learning LM:

* Train a prediction-based neural LM where the NN is a deep model

#### Transformer variations

Transformer proposed for _machine translation_ in a sequence-to-sequence (text in, text out) setting

Two major components:

* ENCODER which interprets text in source language
* DECODER which generates text in target language

BOTH of these are deep stacks of _transformer_ blocks

Three models derived from this architecture:

* Encoder-only models, e.g. BERT
* Decoder-only models, e.g. GPT
* Encoder-decoder models e.g. T5

##### Encoder-only models

Training _Transformer encoder_ as a (birectional) LM produces models that are particularly effective at text classification and sequence labeling tasks

Prototypical example: BERT the original encoder-only model:

* 110M parameter ("base") and 340M parameter ("large") variants
* Trained on corpus of _3B words_ for ~40 epochs (~130B tokens)

Very large number of encoder-only models proposed since, but mostly not much more than 1B parameters
BERT and similar models substantially increased SOTA when introduced and remain very effective tools for classification tasks in NLP

##### Decoder-only models

Training _Transformer decoder_ as a causal ("left-to-right") LM produces models that are particularly effective at generating text. Prototypical example: GPT family

* GPT-1: 117M parameters, GPT-3 175B parameters
* Trained on corpora ranging from 4 to 400GB of text

Large number of _ever-larger_ decoder-only models created, largest of which details are public nearly _2T_ parameters

GPT-like models are the main drivers behind the current generative AI craze.

##### Encoder-decoder models

Traininig the full, original encoder-decoder Transformer architecture as an LM produces effective models fro text-to-text tasks (e.g. translation)

Prototypical example: _T5_:

* Models ranging from _220M_ base model to 11B parameters
* Trained for 34B tokens

Encoder-decoder models are versatile, potentially combining the strengths of both enc-only and dec-only, also as an active area of research, but don't have the ubiquity of encoder-only nor the media attention of decoder-only models.

### Capabilities and limitations of LMs

#### Capabilities

Many capabilities of large LMs emerge with scale. Larger models perform well in tasks that smaller fail. _Prompting_ and examples given as part of the prompt further improve results.

#### Limitations

* LMs hallucinate, can create imagine persons places context so forth.
* LMs can be rude or unreasonable.
* LMs learn and can emphasize biases
  * e.g. racial
  * gender specific work

Very large neural language models trained on "naturally" occuring texts can be _amazingly good_ at predicting words in context (e.g. next word) but such training does not produce:

* Models that can _follow instructions_
* Models that can produce _coherent dialogue_
* Models that are _aligned_ to human intrests (helpful, honest, harmless)

Current efforts in LM development are increasingly focused on these aspects rather than improving "pure" LMs

#### Training LMs to follow instructions

1) STEP: Collect demonstration data and train a supervised policy. A prompt is sampled from our prompt dataset, A labeler demonstrates the desired output behaviour. This data is used to fine-tune GPT-3 with supervised learning
2) Collect comparision data, and train a reward model. A prompt and several model outputs are sampled. A labeler ranks the outputs from best to worst. This data is used to train our reward model
3) Optimize a policy against the reward model using reinforcement learning. A new prompt is sampled from dataset. The policy generates an output. The reward model calculates a reward for the output $r_k$. The reward $r_k$ is used to update the policy using PPO Proximal Policy Optimization.

Fine-tuning with instruction data is basically the same process as pre-training. With sufficient numbers of examples of instructions, inputs and outputs the models can _generalize to new instructions_. Open instruction datasets with millions of examples exist, but _language coverage_ is limited

#### Training LMs to dialogue

Models can be trained to chat like ChatGPT similarly as for following instructions. Additional requirement is that data must include multiple "turns" between prompter and model rather than a single instruction-response cycle.

_High-quality dialogue data_ is even more expensive to create than instruction data - less possibility to build on existing resources.

Instruction fine-tuning makes it more likely that models generate text that is reponsive to instructions.

## Impacts and future

Transformers-based LMs demonstrated _state-of-the-art performance_ across a _very broad range of NLP tasks_ soon after they were introduced. In addition to_single models trained on sigle tasks_ outperforming previously proposed methods, neural LMs have contributed to a major shift towards unification in NLP in may ways:

* _Methodological_: Diverse approaches replaced with few architectures
* _From single- to multi-task_: Task-specific models replaced with general ones
* _Global_: initial progress towards "one model to rule them all"

Historically custom methods for each task, language, and often domain. For example a rule-based parser fro English biomedical text

Since early 2000s, language-independent ML approaches popula but still with

* Task-specific ML methods and often explicit features
* Each combination of task, language and domain requires specific corpus for training and produces a model specific to that combination

Recently _cross-lingual and multitas approaches_ increasingly successful

* Single models can learn multiple tasks and generalize across task boundaries.

With last few years increasing focus on the potential of one model for everything

* Instruction following: "generally intelligent" model can perform _any_ task that can be described with natural language instructions
* Massively multilingual models trained with text in dozens of languages can take input and produce output in any of those languages
* Domain independence models trained with texts representing a broad range of domains, genres, etc. generalize across these

Mention something about cherry-picking and caveats. far from perfect.
