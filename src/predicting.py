# imports
from evaluating import Evaluater

from six.moves import xrange

import numpy as np


# classes
class Predicter(Evaluater):
    def get_output(self, inp, is_batch=False):
        inputs_batch, outputs_batch, target_weights, \
            bucket_id = self._get_inputs(inp, is_batch)

        _, _, out = self.model.step(self.session, inputs_batch,
                                    outputs_batch, target_weights,
                                    bucket_id, True)

        outputs = [[] for _ in xrange(self.model.batch_size)]
        for batch_logits in out:
            for idx, logits in enumerate(batch_logits):
                token_id = int(np.argmax(logits))
                # token = str(self.vocabulary.get_token(token_id))[3:-1]
                token = str(self.vocabulary.get_token(token_id))

                outputs[idx].append(token)

        if is_batch:
            return outputs
        else:
            return outputs[0]
