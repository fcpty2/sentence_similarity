from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class BERTModel:
    def __init__(self):
        self.tokenizer = self.__get_tokenizer()
        self.model = self.__get_model()

    def __get_tokenizer(self):
        return AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    def __get_model(self):
        return AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    # initialize dictionary that will contain tokenized sentences
    def tokenize(self, sentences:list):

        tokens = {'input_ids': [], 'attention_mask': []}

        for sentence in sentences:
            # tokenize sentence and append to dictionary lists
            new_tokens = self.tokenizer.encode_plus(sentence, max_length=128, truncation=True,
                                            padding='max_length', return_tensors='pt')

            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        # reformat list of tensors into single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

        return tokens
    
    def model_setup(self, tokens):
        return self.model(**tokens)

if __name__ == '__main__':
    sentence_list = [
            "Three years later, the coffin was still full of Jello.",
            "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
            "The person box was packed with jelly many dozens of months later.",
            "Standing on one's head at job interviews forms a lasting impression.",
            "It took him a month to finish the meal.",
            "He found a leprechaun in his walnut shell."
    ]

    bert_obj = BERTModel()
    tokens = bert_obj.tokenize(sentences=sentence_list)
    outputs = bert_obj.model_setup(tokens=tokens)

    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']

    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

    ##### Mask embedding #####
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    
    ##### Mean pooling #####
    mean_pooled = summed / summed_mask

    ##### Cosine Similarity #####
    mean_pooled = mean_pooled.detach().numpy()

    sim_score = cosine_similarity(
                [mean_pooled[0]],
                mean_pooled[1:]
            )
    print(sim_score.flatten().tolist())

    print(f"Sentence : {sentence_list[0]} \n")
    pd.DataFrame({'Sentence': sentence_list[1:],
                'SimilarityScore': sim_score.flatten()}).sort_values("SimilarityScore", ascending=False)