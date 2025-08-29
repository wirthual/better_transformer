from transformers import AutoModel
from better_transformer import BetterTransformer



def test1():

    model_id = "BAAI/bge-small-en-v1.5"
    
    model = AutoModel.from_pretrained(model_id)

    model = BetterTransformer.transform(model)

