import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingToSIFNonlinearModel(nn.Module):

    def __init__(self, embedding_size, hidden_size=128, num_outputs=1, min_output=None, max_output=None):
        super(EmbeddingToSIFNonlinearModel, self).__init__()
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        #self.min_output = min_output
        #self.max_output = max_output
        if min_output and max_output:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False
        #self.linear3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
            #x = torch.clamp(x, min=self.min_output, max=self.max_output)
 
        #x = F.relu(x)
        #x = self.linear3(x)
        return x

