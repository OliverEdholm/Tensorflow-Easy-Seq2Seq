# imports
from evaluating import Evaluater


# classes
class Embedder(Evaluater):
    def get_embedding(self, inp, is_batch=False):
        inputs_batch, outputs_batch, target_weights, \
            bucket_id = self._get_inputs(inp, is_batch)

        embeddings = self.model.step(self.session, inputs_batch, outputs_batch,
                                     target_weights, bucket_id, True,
                                     get_embedding=True)

        if is_batch:
            return embeddings[-1]
        else:
            return embeddings[-1][0]
