import torch
import torch.nn as nn


class EmbeddingToSIFModel(nn.Module):

    def __init__(self, embedding_size, num_outputs=1):
        self.linear1 = nn.Linear(embedding_size, num_outputs)

    def forward(self, x):
        out = self.linear1(x)
        return out
