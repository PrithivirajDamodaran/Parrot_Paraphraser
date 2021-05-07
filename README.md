

# Parrot

## 1. What is Parrot?
Parrot is a paraphrase based utterance augmentation framework purpose built to accelerate training NLU models. A paraphrase framework is more than just a paraphrasing model.

## 2. Why Parrot?
**Huggingface** lists [12 paraphrase models,](https://huggingface.co/models?pipeline_tag=text2text-generation&search=paraphrase)  **RapidAPI** lists 7 fremium and commercial paraphrasers like [QuillBot](https://rapidapi.com/search/paraphrase?section=apis&page=1), Rasa has discussed an experimental paraphraser for augmenting text data [here](https://forum.rasa.com/t/paraphrasing-for-nlu-data-augmentation-experimental/27744), Sentence-transfomers offers a [paraphrase mining utility](https://www.sbert.net/examples/applications/paraphrase-mining/README.html) and [NLPAug](https://github.com/makcedward/nlpaug) offers word level augmentation with a [PPDB](http://paraphrase.org/#/download) (a multi-million paraphrase database). While these attempts at paraphrasing are great, there are still some gaps and paraphrasing is NOT yet a mainstream option for text augmentation in building NLU models....Parrot is a humble attempt to fill some of these gaps.

**What is a good paraphrase?** Almost all conditioned text generation models are validated  on 2 factors, (1) if the generated text conveys the same meaning as the original context (Adequacy) (2) if the text is fluent / grammatically correct english (Fluency). For instance Neural Machine Translation outputs are tested for Adequacy and Fluency. But [a good paraphrase](https://www.aclweb.org/anthology/D10-1090.pdf) should be adequate and fluent while being as different as possible on the surface lexical form. With respect to this definition, the  **3 key metrics** that measures the quality of paraphrases are:
 - **Adequacy** (Is the meaning preserved adequately?) 
 - **Fluency** (Is the paraphrase fluent English?) 
 - **Diversity (Lexical / Phrasal / Syntactical)** (How much has the paraphrase changed the original sentence?)

*Parrot offers knobs to control Adequacy, Fluency and Diversity as per your needs.*

**What makes a paraphraser a good augmentor?** For training a NLU model we just don't need a lot of utterances but utterances with intents and slots/entities annotated. Typical flow would be:
- Given an **input utterance  + input annotations** a good augmentor spits out N **output paraphrases** while preserving the intent and slots. 
 - The output paraphrases are then converted into annotated data using the input annotations that we got in step 1.
 - The annotated data created out of the output paraphrases then makes the training dataset for your NLU model.

But in general being a generative model paraphrasers doesn't guarantee to preserve the slots/entities. So the ability to generate high quality paraphrases in a constrained fashion without trading off the intents and slots for lexical dissimilarity makes a paraphraser a good augmentor. *More on this in section 3 below*

### Installation
```python
pip install git+https://github.com/PrithivirajDamodaran/Parrot.git
```

### Quickstart
```python

# Caveat: the generate part is NOT seeded, so for a same input, multiple runs will produce DIFFERENT outputs for now

from parrot import Parrot
import warnings
warnings.filterwarnings("ignore")

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)

phrases = ["Can you recommed some upscale restaurants in Rome?",
           "What are the famous places we should not miss in Russia?"
]

for phrase in phrases:
  print("-"*100)
  print("Input_phrase: ", phrase)
  print("-"*100)
  para_phrases = parrot.augment(input_phrase=phrase)
  for para_phrase in para_phrases:
   print(para_phrase)
```

<pre>
-----------------------------------------------------------------------------
Input_phrase: Can you recommed some upscale restaurants in Rome?
-----------------------------------------------------------------------------
"which upscale restaurants are recommended in rome?"
"which are the best restaurants in rome?"
"are there any upscale restaurants near rome?"
"can you recommend a good restaurant in rome?"
"can you recommend some of the best restaurants in rome?"
"can you recommend some best restaurants in rome?"
"can you recommend some upscale restaurants in rome?"
-----------------------------------------------------------------------------
Input_phrase: What are the famous places we should not miss in Russia
-----------------------------------------------------------------------------
"which are the must do places for tourists to visit in russia?"
"what are the best places to visit in russia?"
"what are some of the most visited sights in russia?"
"what are some of the most beautiful places in russia that tourists should not miss?"
"which are some of the most beautiful places to visit in russia?"
"what are some of the most important places to visit in russia?"
"what are some of the most famous places of russia?"
"what are some places we should not miss in russia?"
</pre>

### Knobs

```python

 para_phrases = parrot.augment(input_phrase=phrase, 
                               diversity_ranker="levenshtein",
                               do_diverse=False, 
                               max_return_phrases = 10, 
                               max_length=32, 
                               adequacy_threshold = 0.99, 
                               fluency_threshold = 0.90)

```


## 3. Scope

In the space of conversational engines, knowledge bots are to which **we ask questions** like *"when was the Berlin wall teared down?"*, transactional bots are to which **we give commands** like *"Turn on the music please"* and voice assistants are the ones which can do both answer questions and action our commands. Parrot mainly foucses on augmenting texts typed-into or spoken-to conversational interfaces for building robust NLU models. (*So usually people neither type out or yell out long paragraphs to conversational interfaces. Hence the pre-trained model is trained  on text samples of maximum length of 64.*)

*While Parrot predominantly aims to be a text augmentor for building good NLU models, it can also be used as a pure-play paraphraser.*


## 4. What makes a paraphraser a good augmentor for NLU? (Details)

To enable automatic training data generation, a paraphraser needs to keep the slots in intact. So the end to end process can take input utternaces, augment and convert them into NLU training format goo et al or rasa format (as shown below). The data generation process needs to look for the same slots in the output paraphrases to derive the start and end positions.(as shown in the json below)

<img src="./images/NLU Flow.png" alt="" title="" width="550" height="100" /> 

Ideally the above process needs an UI like below to collect to input utternaces along with annotations (Intents, Slots and slot types) which then can be agumented and converted to training data.

<img src="./images/Augmentor UI.png" alt="" title="" width="550" height="100" /> 


### Sample NLU data (Rasa format)

```json
{
    "rasa_nlu_data": {
        "common_examples": [
            {
                "text": "i would like to find a flight from charlotte to las vegas that makes a stop in st. louis",
                "intent": "flight",
                "entities": [
                    {
                        "start": 35,
                        "end": 44,
                        "value": "charlotte",
                        "entity": "fromloc.city_name"
                    },
                    {
                        "start": 48,
                        "end": 57,
                        "value": "las vegas",
                        "entity": "toloc.city_name"
                    },
                    {
                        "start": 79,
                        "end": 88,
                        "value": "st. louis",
                        "entity": "stoploc.city_name"
                    }
                ]
            },
            ...
        ]
    }
}
```

 - **Original**:  I would like a list of round trip flights between indianapolis and orlando florida for the 27th
 - **Paraphrase useful for augmenting**: what are the round trip flights between indianapolis and orlando for the 27th
 - **Paraphrase not-so-useful for augmenting**: what are the round trip flights between chicago and orlando for the 27th.


## 5.Pretrained model

 - MSRP Paraphrase 
 - Google PAWS 
 - ParaNMT 
 - Quora question pairs. 
 - SNIPS Alexa commands
 - MSRP Frames
 - GYAFC Dataset

###  Metrics and Comparison
TBD

### Current Features
TBD

### Roadmap
TBD
