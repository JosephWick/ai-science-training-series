import torch
import numpy as np
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import sys
np.set_printoptions(threshold=sys.maxsize)

# Load BERT.
model = BertModel.from_pretrained('bert-large-uncased-whole-word-masking')
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

# Save vocab to file
tokenizer.save_vocabulary(save_directory='.')

print("vocab size:", model.config.vocab_size)

word_embs = model.get_input_embeddings()

# Convert the vocabulary embeddings to numpy.
idxs = np.arange(0,model.config.vocab_size,1)
idxs_input = torch.LongTensor(idxs)
bert_word_embs = word_embs(idxs_input).detach().numpy()
print(bert_word_embs.shape)


# Read in the vocabulary
filename = "vocab.txt"
with open(filename,'r') as f:
    bert_words = np.array([])
    for line in f:
        bert_words = np.append(bert_words, line.rstrip())

bert_char_indices_to_use = np.arange(999, 1063, 1)
bert_voc_indices_to_plot = np.append(bert_char_indices_to_use, np.arange(1996, 5932, 1))
bert_voc_indices_to_use = np.append(bert_char_indices_to_use, np.arange(1996, 11932, 1))

bert_voc_indices_to_use_tensor = torch.LongTensor(bert_voc_indices_to_use)
bert_word_embs_to_use = word_embs(bert_voc_indices_to_use_tensor).detach().numpy()
bert_words_to_plot = bert_words[bert_voc_indices_to_plot]

print(len(bert_voc_indices_to_plot))
print(len(bert_voc_indices_to_use))


# Run tSNE on the BERT vocab
mytsne_words = TSNE(n_components=2,early_exaggeration=12,verbose=2,metric='cosine',init='pca',n_iter=2500)
bert_word_embs_to_use_tsne = mytsne_words.fit_transform(bert_word_embs_to_use)

# save the BERT words to use to file
np.save("tsne_output.npy", bert_word_embs_to_use_tsne, allow_pickle=True)
np.save("berts_to_plot.npy", bert_words_to_plot, allow_pickle=True)
np.save("berts_tsne.npy", bert_word_embs_to_use_tsne, allow_pickle=True)