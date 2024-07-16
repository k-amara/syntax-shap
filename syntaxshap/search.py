
import faiss
import numpy as np
import time
from utils._general import get_embedding_mask

class SimilaritySearch():

    def __init__(self, tokenizer, padding_length, seed, device):
        self.seed = seed
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.device = device


    def __call__(self, embeddings, filtered_ids, embedding_data, args):
        
        d = embeddings.shape[-1]
        
        mask, query_mask = get_embedding_mask(self.tokenizer, embedding_data, self.padding_length, filtered_ids)
        xb = embeddings[query_mask]

        if args.similarity_metric == "euclidean":
            index = faiss.IndexFlatL2(d)
            index.add(embeddings[mask])
        elif args.similarity_metric == "cosine":
            index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
            embedding_vectors = embeddings[mask]
            embedding_vectors = embedding_vectors.detach().cpu().numpy()
            print(embedding_vectors.dtype, embedding_vectors.__class__, embedding_vectors.shape)
            faiss.normalize_L2(embedding_vectors)
            index.add(embedding_vectors)
            xb = xb.detach().cpu().numpy()
            faiss.normalize_L2(xb)

        
        #num_tokens = hidden_indices_sample.sum(axis=1)
        #tokens_cumulative = num_tokens.cumsum()

        k = len(embeddings[mask])
        print(k)
        step_size = 500
        #D = np.zeros((len(xb), k), dtype=np.float32)
        num_query = len(xb)
        I = np.zeros((num_query, k), dtype=np.int32)
        start_time = time.time()
        for i in range(0, num_query, step_size):
            end_index = min(i + step_size, num_query)
            I[i:end_index] = index.search(xb[i:end_index], k)[1].astype(np.int32)
        print(time.time() - start_time)
        #print(D)
        #print(I)

        print(I.shape)
        
        return None, I
