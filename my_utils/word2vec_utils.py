import numpy as np
from .tokenizer import normalize_text
from sklearn.decomposition import PCA

def load_glove_vocab(path, glove_dim, wv_dim=300):
    vocab = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-wv_dim]))
            vocab.add(token)
    return vocab

def build_embedding(path, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, wv_dim))
    emb[0] = 0

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(path, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]

    v = emb
    # Implement all-but-the-top
    mu = np.mean(v, axis=0)
    v_tilde = v - mu  

    D=3
    # Compute the PCA components
    pca = PCA(n_components=D)
    u = pca.fit_transform(v.T)

    # Postprocess
    for w in range(v_tilde.shape[0]):
        for i in range(D):
            v_tilde[w, :] = v_tilde[w, :] - u[:, i].dot(v[w]) * u[:, i].T

    return v_tilde
