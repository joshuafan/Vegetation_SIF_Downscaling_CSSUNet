import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingToSIFNonlinearModel(nn.Module):

    def __init__(self, embedding_size, hidden_size=128, num_outputs=1):
        super(EmbeddingToSIFNonlinearModel, self).__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
