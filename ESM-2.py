import torch
import esm

# Load model and alphabet once globally (so you don't reload every call)
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

def get_sequence_embedding(sequence: str) -> torch.Tensor:
    data = [("protein", sequence)]
    labels, strs, tokens = batch_converter(data)
    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)
    token_embeddings = results["representations"][33]
    sequence_embedding = token_embeddings[0, 1:-1].mean(dim=0)
    return sequence_embedding

if __name__ == "__main__":
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    embedding = get_sequence_embedding(seq)
    print(f"Embedding shape: {embedding.shape}")
