

# Parrot
A practical and feature-rich paraphraser to augment human intents in text form to build robust NLU models for conversational engines.

## Why Parrot?
**Huggingface** [lists 12 paraphrase models,](https://huggingface.co/models?pipeline_tag=text2text-generation&search=paraphrase)  **RapidAPI** [lists 7 fremium / commercial paraphrasers like QuillBot](https://rapidapi.com/search/paraphrase?section=apis&page=1), Rasa [has discussed an experimental paraphraser for augmenting text data](https://forum.rasa.com/t/paraphrasing-for-nlu-data-augmentation-experimental/27744). While these attempts at paraphrasing are great, there are still some gaps and paraphrasing is NOT yet a mainstream option for text augmentation in building NLU models.. Parrot is a humble attempt to fill these gaps.

What is a good paraphrase ? [A good paraphrase should convey the same meaning as the original sentence in fluent and grammtically correct english, while being as different as possible on the surface form](https://www.aclweb.org/anthology/D10-1090.pdf). With respect to this definition, there are **3 key metrics** that measures the quality of paraphrases:

 - **Adequacy** (Is the meaning preserved adequately?) 
 - **Fluency** (Is the paraphrase fluent English?) 
 - **Diversity or Lexical Dissimilarity** (How much has the paraphrase changed the original sentence?)

*Parrot offers knobs to control Adequacy, Fluency and Diversity for your needs.*

**A good paraphraser**
For instance, the below example preserves adequacy (in NLU context: the intent and slots are intact), fluency (grammar is fine) and offers some diversity as for as the utternace is concerned.

 - **Original**:  I would like a list of round trip flights between indianapolis and orlando florida for the 27th
 - **Paraphrase**: what are the round trip flights between indianapolis and orlando for the 27th

**A good augmentor**
While the above example is the strict expectation from a pure-play paraphraser, text augmenting offers some liberty.  As long as the augmentor retains the intent and grammar (with some diversity) in the paraphrases, it is acceptable even if the slots aren't intact. For instance, the below paraphrase is acceptable in an augmentor setting.

 - **Original**:  I would like a list of round trip flights between indianapolis and orlando florida for the 27th
 - **Paraphrase**: what are the round trip flights between chicago and orlando for the 3rd

**A bad augmentor**

 - **Original**:  I would like a list of round trip flights between indianapolis and orlando florida for the 27th
 - **Paraphrase**: what are the round trip flights between chicago and orlando for the 3rd

*While Parrot is predominantly aims to be a text augmentor for building good NLU models, it can also be used as a pure-play paraphraser.*


## Scope
In the space of conversational engines, knowledge bots are to which **we ask questions** like *"when was the berlin wall teared down?"*, transactional bots are to which **we give commands** like *"turn on the music please"* and voice assistants are the ones which can do both answer questions and action our commands. Parrot mainly foucses on augmenting texts typed or spoken to conversational interfaces for building robust NLU models. Hence the pretrained model is trained  on text samples of *maximum length of 64.*

## Installation
```python
pip install parrot
```

## Quickstart
```python
 import Parrot
 import pandas as pd
 pd.set_option('max_colwidth', -1)
 parrot = Parrot("prithivida/parrot_paraphraser_T5","cuda:0")
 phrase = ""
 df = parrot.augment(input_phrase = phrase)
 df.head(df.shape[0])
```

### Pretrained model

### Current Features

### Roadmap
