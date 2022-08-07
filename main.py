import gc
import kfserving
import logging
import math
import os
import signal
import sys
import time

from transformers import AutoConfig, AutoTokenizer
from transformers import TFAutoModelForCausalLM

import torch
import tensorflow as tf

SERVER_NUM_WORKERS = int(os.environ.get('SERVER_NUM_WORKERS', 1))
SERVER_PORT = int(os.environ.get('SERVER_PORT', 8080))
MODEL_DEVICE = 0
MODEL_PATH = '/mnt/models'
MODEL_NAME = os.environ.get('MODEL_NAME', 'GPT-J-6B-lit-v2')
MODEL_FILENAME = os.environ.get('MODEL_FILENAME', 'gpt_lit_v2_rev1.pt')
MODEL_PRECISION = os.environ.get('MODEL_PRECISION', 'native').lower()
READY_FLAG = '/tmp/ready'
DEBUG_MODE = bool(os.environ.get('DEBUG_MODE', 0))

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)
logger = logging.getLogger(MODEL_NAME)


class KFServingHuggingFace(kfserving.KFModel):
    def __init__(self, name):
        super().__init__(name)
        self.name = MODEL_NAME
        self.ready = False
        self.config = None
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.bad_words_ids = None
        self.generate = None

    def load_config(self):
        logger.info(f'Loading config from {MODEL_PATH}')
        self.config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
        logger.info('Config loaded.')

    def load_tokenizer(self):
        logger.info(f'Loading tokenizer from {MODEL_PATH} ...')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self.tokenizer.pad_token_id = ['<|endoftext|>']
        assert self.tokenizer.pad_token_id == 50256, 'incorrect padding token'
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        logger.info('Tokenizer loaded.')

    def load_bad_word_ids(self):
        logger.info('loading bad word ids')

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            add_prefix_space=True
        )

        forbidden = [
            'nigger', 'nigga', 'negro', 'blacks',
            'rapist', 'rape', 'raping', 'niggas', 'raper',
            'niggers', 'rapers', 'niggas', 'NOOOOOOOO',
            'fag', 'faggot', 'fags', 'faggots']

        bad_words_ids = []
        for word in forbidden:
            bad_words_ids.append(tokenizer(word).input_ids)
        self.bad_words_ids = bad_words_ids

        logger.info('done loading bad word ids')

    def load(self):
        """
        Load from a pytorch saved pickle to reduce the time it takes
        to load the model.  To benefit from this it is important to
        have run pytorch save on the same machine / hardware.
        """

        gc.disable()
        start_time = time.time()

        self.load_config()
        self.load_tokenizer()
        self.load_bad_word_ids()

        logger.info(
            f'Loading model from {MODEL_PATH} into device {MODEL_DEVICE}:{torch.cuda.get_device_name(MODEL_DEVICE)}')

        self.model = TFAutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, MODEL_FILENAME), from_pt=True)
        # self.model = torch.load(os.path.join(MODEL_PATH, MODEL_FILENAME))
        self.model.config.eos_token_id = 198
        self.model.config.exponential_decay_length_penalty = None
        self.model.eos_token_id = 198
        self.generate = tf.function(model.generate, jit_compile=True)

        logger.info('Model loaded.')
        # self.model

        logger.info('Creating generator for model ...')
        logger.info(f'Model is ready in {str(time.time() - start_time)} seconds.')

        gc.enable()
        self.ready = True
        self._set_ready_flag()

    def explain(self, request):
        text = request['input_text']
        output_text = request['output_text']
        args = request['parameters']

        input_tokens = self.tokenizer(text, return_tensors="pt")['input_ids'].to(0)
        output_tokens = self.tokenizer(output_text, return_tensors="pt")['input_ids'].to(0)

        args['return_dict_in_generate'] = True
        args['output_scores'] = True
        args['max_new_tokens'] = 1

        logprobs = []
        tokens = []
        for token in output_tokens[0]:
            output = self.model.generate(input_tokens, **args)
            output_probs = torch.stack(output.scores, dim=1).softmax(-1)[0][0]
            prob = output_probs[token]
            logprobs.append(math.log(prob) if prob > 0 else -9999)
            tokens.append(self.tokenizer.decode(token))
            input_tokens = torch.cat((input_tokens, token.resize(1, 1)), dim=1)
        return {'tokens': tokens, 'logprobs': logprobs}

    def predict(self, request, parameters=None):
        # batching requires fixed parameters
        request_params = {
            'temperature': 0.72,
            # 'repetition_penalty': 1.13125,
            'max_new_tokens': 64,
            'top_p': 0.725,
            'top_k': 0,
            'do_sample': True,
            'eos_token_id': 198,
            'bad_words_ids': self.bad_words_ids
        }

        if parameters is not None:
            request_params.update(parameters)

        inputs = request['instances']

        input_ids = self.tokenizer(
            inputs,
            add_special_tokens=False,
            return_tensors="tf",
            return_attention_mask=True,
            padding=True)

        with torch.inference_mode():
            outputs = self.generate(
                input_ids['input_ids'],
                attention_mask=input_ids['attention_mask'],
                **request_params)

        responses = []
        for ins, outs in zip(inputs, outputs):
            decoded = self.tokenizer.decode(
                outs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            decoded = decoded[len(ins):]
            responses.append(decoded.rstrip())

        return {'predictions': responses}

    def _set_ready_flag(self):
        """Used by readiness probe. """
        with open(READY_FLAG, 'w') as fh:
            fh.write('1')


def terminate(signal, frame):
    """
    Kubernetes send SIGTERM to containers for them
    to stop so this must be handled by the process.
    """
    logger.info("Start Terminating")
    if os.path.exists(READY_FLAG):
        os.remove(READY_FLAG)
    time.sleep(5)
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate)

    if DEBUG_MODE:
        import time

        time.sleep(3600 * 10)

    model = KFServingHuggingFace(MODEL_NAME)
    model.load()

    kfserving.KFServer(
        http_port=SERVER_PORT,
        workers=SERVER_NUM_WORKERS
    ).start([model])
