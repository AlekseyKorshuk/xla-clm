import os
import time
from datetime import timedelta
from functools import wraps, partial
from tqdm import tqdm

# JAX imports and settings
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# PyTorch imports and settings
import torch
from transformers.testing_utils import torch_device

torch.backends.cuda.matmul.allow_tf32 = True  # All frameworks using TF32

# TF imports and settings
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Transformers imports last, as they would import the frameworks (and we need to set some options before that)
from transformers import (
    AutoTokenizer, AutoConfig,
    MODEL_FOR_CAUSAL_LM_MAPPING, AutoModelForCausalLM, TFAutoModelForCausalLM, FlaxAutoModelForCausalLM,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, AutoModelForSeq2SeqLM, TFAutoModelForSeq2SeqLM, FlaxAutoModelForSeq2SeqLM,
)

MODEL_NAME = "hakurei/litv2-6B-rev2"  # hakurei/litv2-6B-rev2
RUN_FLAX = False
if RUN_FLAX:
    import jax

GENERATION_KWARGS = {
    "max_new_tokens": 32,
    'eos_token_id': 198,
    'do_sample': False,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    # 'repetition_penalty': 1.13,
}

NUM_RUNS = 10
SAMPLE = GENERATION_KWARGS["do_sample"]
NUM_BEAMS = 1
MAX_NEW_TOKENS = 64

BATCH_SIZE = 1
PAD_MULTIPLE = 1024  # XLA has to retrace for each input length, pads the inputs so as to be multiples of this
TEMPERATURE = 2.0  # used when SAMPLE is True
TOP_K = 50  # used when SAMPLE is True

# Don't check results for generations that are prone to small differences: long generations and many beams.
CHECK_RESULTS = not SAMPLE and (NUM_BEAMS <= 16 and MAX_NEW_TOKENS <= 64)

# different lengths to ensure XLA is able to handle different number of input tokens
INPUT_EXAMPLES = [
    "The dog",
    "Yesterday, there was",
    "Roses are red, Violets are blue",
    "'The quick brown fox jumps over the lazy dog' - he said.",
    "ear's Fright and tried to kill the night guard, who is Michael, Henry or a random unnamed person. "
    "Eventually, the attraction is caught on fire. In the newspaper, Springtrap's head can be seen when brightening up"
    " the image, giving an early hint he survived.\n\nIn the opening scene of Sister Location, an entrepreneur is "
    "asking him questions about the new animatronics. They inquire why certain features were added and express their "
    "concerns, but he avoids answering the specific features they refer to.\n\nHe is also the creator of the Funtime "
    "Animatronics (Assisted by an unknowing Henry) and the former owner of the Circus Baby's Entertainment and Rental, "
    "and, by extension, Circus Baby's Pizza World.\n\nIt's revealed in the final Michael Afton's Cutscene that William "
    "sent his son, Michael, to his rundown factory to find his daughter, but he is 'scooped' as his sister, Baby, "
    "tricked him. Ennard took control over his body, but he manages to survive as Michael becomes a rotting corpse. "
    "He swears to find him.\n\nWilliam Afton returns as the main antagonist. It's revealed that William's old partner, "
    "Henry, lured Springtrap, Scrap Baby (Elizabeth), Molten Freddy (and by extension, the remaining parts of Ennard), "
    "and Lefty (the Puppet) to a new Freddy Fazbear's Pizza. Michael in Freddy Fazbear's Pizzeria Simulator is the "
    "manager. On Saturday, Henry burns the whole pizzeria down, while he dies in the fire. Michael decides to stay in "
    "the fire as well. Springtrap and every other animatronic die in the fire and the souls are free, as their killer "
    "is dead.\n\nWhile not directly appearing, footprints that are very similar to Springtrap's can be found behind "
    "the house in Midnight Motorist's secret minigame, presumably luring away the child of the abusive father in the "
    "game.\n\nSeen when completing the Fruity Maze game, standing next to a girl named Susie from the right is William "
    "Afton wearing the Spring Bonnie suit that he eventually was trapped in and became Springtrap he then seemingly "
    "murders Susie.\nWilliam Afton: ...\nMe: \u2026\nWilliam Afton:"
]


def measure_time(function):
    """ Decorator to print execution time of a function """

    @wraps(function)
    def decorated(inputs):
        start = time.time()
        fn_output = function(inputs)
        end = time.time()
        duration = timedelta(seconds=end - start)
        return fn_output, duration

    return decorated


def get_tokenizer(model_name):
    config = AutoConfig.from_pretrained(model_name)
    if type(config) in MODEL_FOR_CAUSAL_LM_MAPPING:
        tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token="</s>", padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_inputs(tokenizer, index, return_tensors, use_xla=False):
    input_sentences = [INPUT_EXAMPLES[i % len(INPUT_EXAMPLES)] for i in range(index, index + BATCH_SIZE)]
    tokenizer_kwargs = {"return_tensors": return_tensors, "padding": True, "truncation": True}
    if use_xla:
        tokenizer_kwargs.update({"pad_to_multiple_of": PAD_MULTIPLE})
    return tokenizer(input_sentences, **tokenizer_kwargs)


def get_model(framework, model_name):
    config = AutoConfig.from_pretrained(model_name)
    model = None

    if type(config) in MODEL_FOR_CAUSAL_LM_MAPPING:
        if framework == "tf":
            model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
        elif framework == "pt":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif framework == "flax":
            model = FlaxAutoModelForCausalLM.from_pretrained(model_name)
    elif type(config) in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
        if framework == "tf":
            model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif framework == "pt":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif framework == "flax":
            if model_name == "t5-3b":  # no FLAX weights for this one, but it can be loaded from PT weights
                model = FlaxAutoModelForSeq2SeqLM.from_pretrained(model_name, from_pt=True)
            else:
                model = FlaxAutoModelForSeq2SeqLM.from_pretrained(model_name)

    if model is None:
        raise ValueError(f"Invalid framework ({framework}) / model ({model_name}) combination.")
    return model


def print_status(all_outputs, all_durations):
    print(f"Execution time -- 1st call: {all_durations[0] / 1000:.2f} ms")
    all_durations = all_durations[1:]
    print(f"Execution time -- mean: {(sum(all_durations) / len(all_durations)) / 1000:.2f} ms")
    all_outputs = all_outputs[1:]
    try:
        mean_length = sum([out.shape[1] for out in all_outputs]) / len(all_outputs)
    except:
        mean_length = sum([out.sequences.shape[1] for out in all_outputs]) / len(all_outputs)
    print(f"Outputs -- mean length: {mean_length:.2f} tokens")


def main_tf_eager():
    model = get_model("tf", MODEL_NAME)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = get_tokenizer(MODEL_NAME)

    @measure_time
    def _generate(inputs):
        # inputs.update({"do_sample": SAMPLE, "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS})
        # if SAMPLE:
        #     inputs.update({"temperature": TEMPERATURE, "top_k": TOP_K})
        return model.generate(**inputs, **GENERATION_KWARGS)

    all_durations = []
    all_outputs = []
    for i in tqdm(range(int(NUM_RUNS / 10))):  # TF Eager is very slow :(
        inputs = get_inputs(tokenizer, index=i, return_tensors="tf")
        gen_out, duration = _generate(inputs)
        all_durations.append(duration.microseconds + duration.seconds * 1e6)
        all_outputs.append(gen_out)
    print_status(all_outputs, all_durations)
    return all_outputs


def main_tf_xla():
    model = get_model("tf", MODEL_NAME)
    model.config.pad_token_id = model.config.eos_token_id
    tokenizer = get_tokenizer(MODEL_NAME)

    xla_generate = tf.function(model.generate, jit_compile=True)

    @measure_time
    def _generate(inputs):
        # inputs.update({"do_sample": SAMPLE, "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS})
        # if SAMPLE:
        #     inputs.update({"temperature": TEMPERATURE, "top_k": TOP_K})
        return xla_generate(**inputs, **GENERATION_KWARGS)

    inputs = get_inputs(tokenizer, index=0, return_tensors="tf", use_xla=True)
    _, _ = _generate(inputs)

    all_durations = []
    all_outputs = []
    for i in tqdm(range(NUM_RUNS)):
        inputs = get_inputs(tokenizer, index=i, return_tensors="tf", use_xla=True)
        gen_out, duration = _generate(inputs)
        all_durations.append(duration.microseconds + duration.seconds * 1e6)
        all_outputs.append(gen_out)
    print_status(all_outputs, all_durations)
    return all_outputs


def main_pt():
    model = get_model("pt", MODEL_NAME)
    model.config.pad_token_id = model.config.eos_token_id
    model.to(torch_device)
    tokenizer = get_tokenizer(MODEL_NAME)

    @measure_time
    def _generate(inputs):
        with torch.no_grad():
            # inputs.update({"do_sample": SAMPLE, "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS})
            # if SAMPLE:
            #     inputs.update({"temperature": TEMPERATURE, "top_k": TOP_K})
            return model.generate(**inputs, **GENERATION_KWARGS)

    inputs = get_inputs(tokenizer, index=0, return_tensors="pt", use_xla=False)
    inputs.to(torch_device)
    _, _ = _generate(inputs)

    all_durations = []
    all_outputs = []
    for i in tqdm(range(NUM_RUNS)):
        inputs = get_inputs(tokenizer, index=i, return_tensors="pt")
        inputs.to(torch_device)
        gen_out, duration = _generate(inputs)
        all_durations.append(duration.microseconds + duration.seconds * 1e6)
        all_outputs.append(gen_out)
    print_status(all_outputs, all_durations)
    del model
    torch.cuda.empty_cache()
    return all_outputs


def main_flax():
    model = get_model("flax", MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    model.config.pad_token_id = model.config.eos_token_id

    generate_kwargs = {"do_sample": SAMPLE, "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS}
    if SAMPLE:
        generate_kwargs.update({"temperature": TEMPERATURE, "top_k": TOP_K})
    jit_generate = jax.jit(partial(model.generate, **generate_kwargs))

    @measure_time
    def _generate(inputs):
        return jit_generate(**inputs)

    all_durations = []
    all_outputs = []
    for i in tqdm(range(NUM_RUNS)):
        inputs = get_inputs(tokenizer, index=i, return_tensors="np", use_xla=True)
        gen_out, duration = _generate(inputs)
        all_durations.append(duration.microseconds + duration.seconds * 1e6)
        all_outputs.append(gen_out)
    print_status(all_outputs, all_durations)
    return all_outputs


def check_outputs(pt_out, eager_out, xla_out, flax_out):
    if CHECK_RESULTS:
        print("\n\nCHECKING OUTPUTS")
        tokenizer = get_tokenizer(MODEL_NAME)
        num_sentences = len(INPUT_EXAMPLES)
        pt_decoded = [
            tokenizer.decode(out[0, :], skip_special_tokens=True) for i, out in enumerate(pt_out) if i <= num_sentences
        ]
        # eager_decoded = [
        #     tokenizer.decode(out[0, :], skip_special_tokens=True) for i, out in enumerate(eager_out) if i <= num_sentences
        # ]
        xla_decoded = [
            tokenizer.decode(out[0, :], skip_special_tokens=True) for i, out in enumerate(xla_out) if i <= num_sentences
        ]
        if flax_out:
            flax_decoded = [
                tokenizer.decode(out.sequences[0, :], skip_special_tokens=True) for i, out in enumerate(flax_out) if
                i <= num_sentences
            ]
        failed_checks = False
        for i in range(num_sentences):
            failed_this_check = False
            # if pt_decoded[i] != eager_decoded[i]:
            #     print(f"FAILED: pt_decoded[{i}] != eager_decoded[{i}]")
            #     failed_this_check = True
            if pt_decoded[i] != xla_decoded[i]:
                print(f"FAILED: pt_decoded[{i}] != xla_decoded[{i}]")
                failed_this_check = True
            if flax_out:
                if pt_decoded[i] != flax_decoded[i]:
                    print(f"FAILED: pt_decoded[{i}] != flax_decoded[{i}]")
                    failed_this_check = True
            if failed_this_check:
                print(f"PT    : {pt_decoded[i]}")
                # print(f"EAGER : {eager_decoded[i]}")
                print(f"XLA   : {xla_decoded[i]}")
                if flax_out:
                    print(f"FLAX  : {flax_decoded[i]}")
                print("")
            failed_checks |= failed_this_check
        if not failed_checks:
            print("All outputs match")


if __name__ == "__main__":
    if RUN_FLAX:
        print("\n\nFLAX")
        flax_out = main_flax()
    else:
        flax_out = None
    print("\n\nPYTORCH")
    pt_out = main_pt()
    # print("\n\nTF (NO XLA)")
    # eager_out = main_tf_eager()
    eager_out = None
    print("\n\nTF (XLA)")
    xla_out = main_tf_xla()
    check_outputs(pt_out, eager_out, xla_out, flax_out)
