import torch
import pickle as pkl
import os
import faiss
import numpy as np

from datasets import load_data
from model import load_model
from utils import arg_parse, fix_random_seed
from utils._general import replace_token
from tqdm import tqdm


class Embeddings:

    def __init__(self, model, tokenizer, device, args):

        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.padding_length = args.padding_length
        if args.model_name == "gemma-2b":
            self.hidden_size = 2048
        elif args.model_name == "mistral":
            self.hidden_size = 4096
        self.batch_size = args.batch_size
        self.hidden_states_dir = args.hidden_states_dir
        self.layer = args.embedding_layer
        self.device = device


    def dump_hidden_states(self, hidden_states, sentence_id):
        with open(os.path.join(self.hidden_states_dir, f"hidden_states_sentence_{sentence_id}_{self.layer}_tokenized.pkl"), "wb") as f:
            pkl.dump(hidden_states, f)
    
    
    def __call__(self, input_ids, sentence_id):
        """
        Given a list of the same sentence 
        input_ids: list of the token ids in the sentence_id (n_tokens)
        
        Return: embeddings (n_token * voc_size * embeddings)
        """
        assert self.layer in ["upper", "lower"], "`layer` must be either 'upper' or 'lower'"
        hidden_path = os.path.join(self.hidden_states_dir, f"hidden_states_sentence_{sentence_id}_{self.layer}_tokenized.pkl")
        if os.path.exists(hidden_path):
            print("Loading pickled hidden_layer...")
            with open(hidden_path, "rb") as f:
                hidden_states = pkl.load(f)
        else:
            os.makedirs(self.hidden_states_dir)
            if self.layer == "upper":
                layer_id = -2
            elif self.layer == "lower":
                layer_id = 1
            # create empty tensor to store hidden states
            hidden_states = torch.zeros((len(input_ids), self.vocab_size, self.hidden_size))
            
            with torch.inference_mode():
                for k, token in enumerate(input_ids):
                    replaced_input_ids = replace_token(input_ids, k, self.vocab_size, special_tokens=[1])
                    outputs = self.model(replaced_input_ids, output_hidden_states=True)
                    # check that outputs.hidden_states[layer_id] has size n_tokens * hidden_size
                    print(outputs.hidden_states[layer_id].size())
                    hidden_states[k] = outputs.hidden_states[layer_id][k] # a list of the embeddings of each token in the vocabulary when replacing the token at position k
            # check hidden_states size 
            self.dump_hidden_states(hidden_states, sentence_id)
        return hidden_states
    
    

class RankSearch():

    def __init__(self, tokenizer, args):
        self.seed = args.seed
        self.tokenizer = tokenizer
        self.device = args.device
        self.ranks_dir = os.path.join(args.result_save_dir, "ranks")
        
    def save_ranks(self, ranks, sentence_id):
        with open(os.path.join(self.hidden_states_dir, f"ranks_sentence_{sentence_id}.pkl"), "wb") as f:
            pkl.dump(ranks, f)


    def __call__(self, embeddings, query_token_ids, sentence_id, args):
        """
        embeddings (n_token * size_voc * hidden_size): the embeddings of each token 
                                                    in the vocabulary when replacing each token in the sentence.
        query_token_ids: the token ids of the sentence
        sentence_id
        """
        
        num_query_token, size_voc, hidden_size = embeddings.shape
        # Initialize an array to store the extracted embeddings
        query_embeddings = np.zeros((len(query_token_ids), hidden_size))

        for i, token_id in enumerate(query_token_ids):
            # Extract the embedding for the given token_id
            query_embeddings[i] = embeddings[i, token_id, :]
        query_embeddings = query_embeddings.detach().cpu().numpy()
        
        ranks_sentence_id = np.zeros((num_query_token, size_voc), dtype=np.int32)
        for i, token_id in enumerate(query_token_ids):
            index = faiss.index_factory(hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)
            embs = embeddings[i]
            query_emb = query_embeddings[i]
            faiss.normalize_L2(embs)
            faiss.normalize_L2(query_emb)
            index.add(embs)
            ranks_sentence_id[i] = index.search(query_emb, size_voc)[1].astype(np.int32)
        print(ranks_sentence_id.shape)
        self.save_ranks(ranks_sentence_id, sentence_id)
        
        return ranks_sentence_id



def compute_embeddings():
    # Set random seed
    fix_random_seed(args.seed)
    
    # Determine device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model(device, args)

    # Prepare the data
    data, data_ids = load_data(tokenizer, args)
    print("Length of data:", len(data))
    
    embedding_model = Embeddings(model, tokenizer, device, args)
    
    for i, str_input in enumerate(tqdm(data)):
        # tokenize the str_input 
        inputs = tokenizer(str_input)
        embeddings = embedding_model(inputs["input_ids"], data_ids[i])
        
    



if __name__ == "__main__":
    parser, args = arg_parse()