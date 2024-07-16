import torch
import pickle as pkl
import os


class Embeddings:

    def __init__(self, model, tokenizer, device, args):

        self.model = model
        self.tokenizer = tokenizer
        self.padding_length = args.padding_length
        if args.model_name == "gemma-2b":
            self.hidden_size = 2048
        elif args.model_name == "mistral":
            self.hidden_size = 4096
        self.batch_size = args.batch_size
        self.hidden_states_dir = args.hidden_states_dir
        self.layer = args.embedding_layer
        self.device = device


    def dump_hidden_states(self, hidden_states, sentence_id, token_pos):
        with open(os.path.join(self.hidden_states_dir, f"hidden_states_sentence_{sentence_id}_pos_{token_pos}_{self.layer}_tokenized.pkl"), "wb") as f:
            pkl.dump(hidden_states, f)
    
    
    def __call__(self, input_ids, sentence_id, token_pos):
        """
        Given a list of the same sentence with one token changing 
        input_ids: vocab_size * n_tokens_in_sentence
        """
        assert self.layer in ["upper", "lower"], "`layer` must be either 'upper' or 'lower'"
        hidden_path = os.path.join(self.hidden_states_dir, f"hidden_states_{self.layer}_tokenized.pkl")
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
            hidden_states = torch.zeros((len(input_ids), self.padding_length, self.hidden_size))
            with torch.inference_mode():
                    outputs = self.model(input_ids, output_hidden_states=True)
                    embeddings_at_pos = outputs.hidden_states[layer_id][token_pos]
            self.dump_hidden_states(embeddings_at_pos, sentence_id, token_pos)
        return hidden_states