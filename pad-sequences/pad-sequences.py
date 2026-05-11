import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if seqs is None or len(seqs) == 0:
        return np.array([])
        
    if max_len is None: 
        L = max(len(seq) for seq in seqs)
    else: 
        L = max_len

    N = len(seqs)
    output = np.full((N,L), pad_value)
    
    for i, seq in enumerate(seqs):
        length =  min(len(seq),L)
        output[i, :length] = seq[:length]
        
    return output