import torch
import pickle as pkl
import os


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
    
def replace_token(input_ids, k, vocab_size, special_tokens=[1]):
    replaced_input_ids = []
    for i in range(len(vocab_size)):
        input_ids[k] = i
        replaced_input_ids.append(input_ids)
    # Remove special tokens 
    return replaced_input_ids[~special_tokens]