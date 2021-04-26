
# Parrot
A realistic paraphraser to augment human intents in the form of text data to build robust NLU models for conversational bots.

### Why Parrot?
HF lists n paraphrase models, RapidAPI lists commercial paraphrasers like QuillBot, Rasa has released an experimental paraphrase for augmenting text data. In spite of all these attempts a good robust paraphraser is still 

### Requirements of a Paraphraser
[A good paraphrase should convey the same meaning as the original sentence in fluent and grammtically correct english, while being as different as possible on the surface form](https://www.aclweb.org/anthology/D10-1090.pdf). With respect to this definition, there are **3 key metrics** that measures the quality of paraphrases:

 - **Adequacy** (Is the meaning preserved adequately?) 
 - **Fluency** (Is the paraphrase fluent English?) 
 - **Diversity or Lexical Dissimilarity** (How much has the paraphrase changed the original sentence?)

For instance, the below example preserves adequacy (in NLU context: the intent and slots are intact), fluency (grammar is fine) and offers a decent diversity.  

> **Original**:  I would like a list of round trip flights between indianapolis and orlando florida for the 27th
> **Paraphrase**: what are the round trip flights between indianapolis and orlando for the 27th

While this is the strict expectation from a pure-play paraphrase, text augmenting offers some lineancy.  As long as the augmentor retains the intent and grammar (with some diversity) in the paraphrases, even if the slots aren't intact it is acceptable. For instance, the below paraphrase is acceptable in an augmentor setting.

> **Original**:  I would like a list of round trip flights between indianapolis and orlando florida for the 27th
> **Paraphrase**: what are the round trip flights between chicago and orlando for the 3rd

While Parrot is predominantly aims to be a robust text augmentor for building good NLU models, it can be used as both a pure-play paraphraser and an augmentor. 

### Scope

### Installation

    pip install parrot

### Quickstart
```lang-js
var x = 10
```

### Pretrained model

### Current Features

### Roadmap
