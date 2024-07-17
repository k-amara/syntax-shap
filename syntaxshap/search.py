
import faiss
import numpy as np
import time
import os
import pickle as pkl

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
            index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
            embs = embeddings[i]
            query_emb = query_embeddings[i]
            faiss.normalize_L2(embs)
            faiss.normalize_L2(query_emb)
            index.add(embs)
            ranks_sentence_id[i] = index.search(query_emb, size_voc)[1].astype(np.int32)
        print(ranks_sentence_id.shape)
        self.save_ranks(ranks_sentence_id, sentence_id)
        
        return ranks_sentence_id
