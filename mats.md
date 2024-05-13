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