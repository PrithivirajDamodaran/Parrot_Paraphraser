class Parrot():
  
  def __init__(self, model_tag="./models/parrot_paraphraser_on_T5", use_gpu=False):
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    import pandas as pd
    from filters import Adequacy
    from filters import Fluency
    from filters import Diversity
    self.tokenizer = AutoTokenizer.from_pretrained(model_tag)
    self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_tag)
    if use_gpu:
        device= "cuda:0"
    else:
        device = "cpu"

    self.device    = device
    self.model     = self.model.to(device)
    self.adequacy_score = Adequacy()
    self.fluency_score  = Fluency()
    self.diversity_score= Diversity()

  def rephrase(self, input_phrase, diversity_ranker='levenshtein', do_diverse=False, style=1, max_length=32, adequacy_threshold = 0.90, fluency_threshold = 0.90):
      import re
      input_phrase = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
      input_phrase = "paraphrase: " + input_phrase
      input_ids = self.tokenizer.encode(input_phrase, return_tensors='pt')
      input_ids = input_ids.to(self.device)
      max_return_phrases = 10
      if do_diverse:
        for n in range(2, 9):
          if max_return_phrases % n == 0:
            break 
        print("max_return_phrases - ", max_return_phrases , " and beam groups -", n)            
        preds = self.model.generate(
              input_ids,
              do_sample=False, 
              max_length=max_length, 
              num_beams = max_return_phrases,
              num_beam_groups = n,
              diversity_penalty = 2.0,
              early_stopping=True,
              num_return_sequences=max_return_phrases)
      else: 
        preds = self.model.generate(
                input_ids,
                do_sample=True, 
                max_length=max_length, 
                top_k=50, 
                top_p=0.95, 
                early_stopping=True,
                num_return_sequences=max_return_phrases) 
        
      paraphrases= set()

      for pred in preds:
        gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
        gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
        paraphrases.add(gen_pp)

         

      adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold)
      if len(adequacy_filtered_phrases) > 0 :
        fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold)
        if len(fluency_filtered_phrases) > 0 :
            diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases, diversity_ranker)
            para_phrases = []
            for para_phrase, diversity_score in diversity_scored_phrases.items():
                para_phrases.append((para_phrase, diversity_score))
            para_phrases.sort(key=lambda x:x[1], reverse=True)    
            return para_phrases[0]
        else:
            return [input_phrase]

  def augment(self, input_phrase, diversity_ranker='levenshtein', do_diverse=False, max_return_phrases = 10, max_length=32, adequacy_threshold = 0.90, fluency_threshold = 0.90):
      import re

      input_phrase = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
      input_phrase = "paraphrase: " + input_phrase
      input_ids = self.tokenizer.encode(input_phrase, return_tensors='pt')
      input_ids = input_ids.to(self.device)

      if do_diverse:
        for n in range(2, 9):
          if max_return_phrases % n == 0:
            break 
        print("max_return_phrases - ", max_return_phrases , " and beam groups -", n)            
        preds = self.model.generate(
              input_ids,
              do_sample=False, 
              max_length=max_length, 
              num_beams = max_return_phrases,
              num_beam_groups = n,
              diversity_penalty = 2.0,
              early_stopping=True,
              num_return_sequences=max_return_phrases)
      else: 
        preds = self.model.generate(
                input_ids,
                do_sample=True, 
                max_length=max_length, 
                top_k=50, 
                top_p=0.95, 
                early_stopping=True,
                num_return_sequences=max_return_phrases) 
        

      paraphrases= set()

      for pred in preds:
        gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
        gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
        paraphrases.add(gen_pp)


      adequacy_filtered_phrases = self.adequacy_score.filter(input_phrase, paraphrases, adequacy_threshold)
      if len(adequacy_filtered_phrases) > 0 :
        fluency_filtered_phrases = self.fluency_score.filter(adequacy_filtered_phrases, fluency_threshold)
        if len(fluency_filtered_phrases) > 0 :
            diversity_scored_phrases = self.diversity_score.rank(input_phrase, fluency_filtered_phrases, diversity_ranker)
            para_phrases = []
            for para_phrase, diversity_score in diversity_scored_phrases.items():
                para_phrases.append((para_phrase, diversity_score))
            para_phrases.sort(key=lambda x:x[1], reverse=True)
            return para_phrases
        else:
            return [input_phrase]



