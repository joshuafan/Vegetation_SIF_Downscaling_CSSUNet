
class SubtileSIFModel(nn.Module):
    def __init__(self, tile2vec_model, embedding_to_sif_model):
        super(SubtileSIFModel, self).__init__()
        self.tile2vec_model = tile2vec_model
        self.embedding_to_sif_model = embedding_to_sif_model
 
    def forward(self, x):
        x = self.tile2vec_model(x)
        x = self.embedding_to_sif_model(x)
        return x
