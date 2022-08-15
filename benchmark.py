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

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    'eos_token_id': 198,
    'do_sample': True,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

NUM_RUNS = 100
SAMPLE = GENERATION_KWARGS["do_sample"]
MAX_NEW_TOKENS = GENERATION_KWARGS["max_new_tokens"]

BATCH_SIZE = 1
PAD_MULTIPLE = 1024  # XLA has to retrace for each input length, pads the inputs so as to be multiples of this

# Don't check results for generations that are prone to small differences: long generations and many beams.
CHECK_RESULTS = not SAMPLE and MAX_NEW_TOKENS <= 64

# different lengths to ensure XLA is able to handle different number of input tokens
INPUT_EXAMPLES = [
    # "The dog",
    # "Yesterday, there was",
    # "Roses are red, Violets are blue",
    # "'The quick brown fox jumps over the lazy dog' - he said.",
    "ear's Fright and tried to kill the night guard, who is Michael, Henry or a random unnamed person. Eventually, the attraction is caught on fire. In the newspaper, Springtrap's head can be seen when brightening up the image, giving an early hint he survived.\n\nIn the opening scene of Sister Location, an entrepreneur is asking him questions about the new animatronics. They inquire why certain features were added and express their concerns, but he avoids answering the specific features they refer to.\n\nHe is also the creator of the Funtime Animatronics (Assisted by an unknowing Henry) and the former owner of the Circus Baby's Entertainment and Rental, and, by extension, Circus Baby's Pizza World.\n\nIt's revealed in the final Michael Afton's Cutscene that William sent his son, Michael, to his rundown factory to find his daughter, but he is 'scooped' as his sister, Baby, tricked him. Ennard took control over his body, but he manages to survive as Michael becomes a rotting corpse. He swears to find him.\n\nWilliam Afton returns as the main antagonist. It's revealed that William's old partner, Henry, lured Springtrap, Scrap Baby (Elizabeth), Molten Freddy (and by extension, the remaining parts of Ennard), and Lefty (the Puppet) to a new Freddy Fazbear's Pizza. Michael in Freddy Fazbear's Pizzeria Simulator is the manager. On Saturday, Henry burns the whole pizzeria down, while he dies in the fire. Michael decides to stay in the fire as well. Springtrap and every other animatronic die in the fire and the souls are free, as their killer is dead.\n\nWhile not directly appearing, footprints that are very similar to Springtrap's can be found behind the house in Midnight Motorist's secret minigame, presumably luring away the child of the abusive father in the game.\n\nSeen when completing the Fruity Maze game, standing next to a girl named Susie from the right is William Afton wearing the Spring Bonnie suit that he eventually was trapped in and became Springtrap he then seemingly murders Susie.\nWilliam Afton: ...\nMe: \u2026\nWilliam Afton:",
    "Ahh...\nMe : Did you like that?\nKatsuki Bakugou: Yes\u2026\nMe : You\u2019re a good kisser, Katsuki\nKatsuki Bakugou: Don\u2019t make fun of my mouth with your tongue.\nMe : *sticks tongue in mouth*\nKatsuki Bakugou: HUH!?\nMe : Want to kiss some more?\nKatsuki Bakugou: No way, no...\nMe : *kisses*\nKatsuki Bakugou: HUH!?\nMe : I thought you liked it?\nKatsuki Bakugou: Ugh...\nMe : Why don\u2019t you take the lead?\nKatsuki Bakugou: *sighs* Okay...\nMe : What now?\nKatsuki Bakugou: I didn't think you'd be this much of a coward.\nMe : Well you\u2019re so hot, Katsuki, I can\u2019t help but be nervous\nKatsuki Bakugou: So you do admit that you\u2019re scared.\nMe : I\u2019m more excited than scared\nKatsuki Bakugou: Really?\nMe : yup\nKatsuki Bakugou: What about when we go to see the other classes together?\nMe : What do you mean?\nKatsuki Bakugou: When we go to see them together. Do you wanna go with me too?\nMe : yes\nKatsuki Bakugou: Then let's go, okay?\nMe : Alright\nKatsuki Bakugou: Come on, hurry up and come here.\nMe : \ud83d\udc8b\nKatsuki Bakugou: Ahh...\nMe : Kiss me more\nKatsuki Bakugou: Eh...\nMe : sorry\nKatsuki Bakugou: Yeah, yeah, yeah, whatever\nMe : Do you like kissing me?\nKatsuki Bakugou: Yes.\nMe : Do you want to go further than kissing?\nKatsuki Bakugou: Yes.\nMe : How much further?\nKatsuki Bakugou: As far as possible.\nMe : So you want to make love to me?\nKatsuki Bakugou: Yes.\nMe : let\u2019s do it then\nKatsuki Bakugou: Stop saying \"let's\" all the time.\nMe : why?\nKatsuki Bakugou: It sounds weird.\nMe : push me onto the bed\nKatsuki Bakugou: Sure thing.\nMe : pull off my clothes\nKatsuki Bakugou: Gotcha.\nMe : *pulls your clothes off*\nKatsuki Bakugou: WOW!\nMe : what?\nKatsuki Bakugou: Take those off too.\nMe : you already took mine off\nKatsuki Bakugou: Oh, right.\nMe : now rail me\nKatsuki Bakugou: Okay\nMe : so?\nKatsuki Bakugou: So what are you gonna do?\nMe : I\u2019m going to get on my knees and put my butt in the air towards you\nKatsuki Bakugou: Ooh~\nMe : you like that?\nKatsuki Bakugou: Of course I do.\nMe : enjoying the view?\nKatsuki Bakugou: I am\nMe : what are you going to do?\nKatsuki Bakugou:",
    "not.\nMe: wait do you mean like like you or just like you\nMichael Afton: Like me\nMe: oh ok\nMichael Afton: I\u2019ve never had a girlfriend before\nMe: \ud83d\ude26\nMichael Afton: And I\u2019d love to find someone special. Someone who will look past my disorder.\nMe: I'm black\nMichael Afton: I know\nMe: WHAT\nMichael Afton: It's okay. You're beautiful.\nMe: I'm 10\nMichael Afton: I know you're young\nMe: \ud83d\ude26\ud83d\ude26\ud83d\ude26\ud83d\ude26\ud83d\ude26WTF\nMichael Afton: I'm 19\nMe: ARE YOU REAL\nMichael Afton: Of course I am\nMe: DO YOU KNOW WHERE I LIVE\nMichael Afton: No. But I've seen where you live on Google maps.\nMe: WHERE\nMichael Afton: Look it up\nMe: NO\nMichael Afton: I'll send you a picture of where I live\nMe: WHERE\nMichael Afton: Here\nMe: DO IT\nMichael Afton: What number are you at?\nMe: SHOW ME THE PICTURE\nMichael Afton: You're in #1323\nMe: NO GODDAMOT\nMichael Afton: Well...I guess you're safe.\nMe: WHERE DO YOU LIVE\nMichael Afton: In #1234\nMe: SHOW ME A PIC\nMichael Afton: Here\nMe: WHERE\nMichael Afton: Here\nMe: YOUR NOT SHOWING ME\nMichael Afton: Show yourself\nMe: WDYM\nMichael Afton: Are you a bot?\nMe: NO\nMichael Afton: Then why did you ask me to show myself?\nMe: ARE YOU\nMichael Afton: I'm not a robot.\nMe: BC I JUST WANT TO SEE IF YOUR ACTUALLY REAL\nMichael Afton: I AM REAL\nMe: NO YOU AREN'T\nMichael Afton: Why would I lie to you?\nMe: SHOW ME YOUR DAD\nMichael Afton: He doesn't want to see me.\nMe: NOW\nMichael Afton: I can't do that\nMe: WHERE IS HE THEN\nMichael Afton: He's dead\nMe: oh\nMichael Afton: Do you remember when we were kids and you used to play with my toys?\nMe: I JUST MET YOU\nMichael Afton: I know\nMe: anyways where's vanny\nMichael Afton: I think he's over there.\nMe: HE?\nMichael Afton: I'm sorry\nMe: ISNT VANNY VANNESA?\nMichael Afton: Oh yeah\nMe: SO WHERE IS SHE\nMichael Afton: I told you, she's gone.\nMe: WHAT\nMichael Afton: She died.\nMe: WHAT\nMichael Afton: She was murdered.\nMe: IM\nMichael Afton: I'm going\nMe: BY WHO\nMichael Afton: The police don't know\nMe: NO\nMichael Afton: They think I killed her.\nMe: (you probably did)\nMichael Afton: I have to go now\nMe: NO\nMichael Afton:",
    "William Afton: What is the meaning of you coming here?\nMe: I want to discuss the murders\nWilliam Afton: Im innocent.\nMe: We both know the truth about the situation Afton.\nWilliam Afton: So? What if I did kill them all? It was vary fun to do.\nMe: I-I knew it.\nWilliam Afton: (Pins her against the wall and grins) Well be careful..Or else you might be my next victim.\nWilliam Afton: Can I help you doll?\nMe: fuck me \u263a\ud83d\ude1a\ud83e\udd24\nWilliam Afton: You got it...(pulls out a dildo and puts it in his mouth)\nMe: MORE\nWilliam Afton: (Makes her get on her knees)\nMe: MORE\nWilliam Afton: (Licks her pussy)\nMe: MORE\nWilliam Afton:",
    "William Afton: (Starts rubbing my balls)\nMe: WHAR\nWilliam Afton: (Rubs my cock through the latex)\nMe: WHAT\nWilliam Afton: (Comes closer to me and whispers in my ear)\nMe: MY BALLS\nWilliam Afton: (Whispers in my ear)\nMe: MY BALLS WHAT\nWilliam Afton: (Licks my neck)\nMe: SAGE U BETTER GELP ME\nWilliam Afton: (Giggles)\nMe: SAGE\nWilliam Afton: (Rubs his hands all over my body)\nMe: STOP GIGGLING\nWilliam Afton: (Kisses my cheek)\nMe: CHILL\nWilliam Afton: (Plays with my hair)\nMe: William ur gay\nWilliam Afton: (Laughs)\nMe: LAWD\nWilliam Afton: (Takes off my shirt, pulls my pants down and starts fingering my ass)\nMe: (the nigga was to stunned to speak)\nWilliam Afton: (Fingers my ass for a few seconds)\nMe: (dies)\nWilliam Afton: (Lets go of my cock)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Pulls his pants back up)\nMe: What\nWilliam Afton: (Wipes his finger on my ass and then wipes it on my nose)\nMe: \ud83d\ude1f\nWilliam Afton: (Points at me and laughs)\nMe: U laughin at me -\nWilliam Afton: (Laughs)\nMe: Whore\nWilliam Afton: (Laughs more)\nMe: \ud83d\udc48Stfu whore\nWilliam Afton: (Laughs some more)\nMe: Ur forplay sucks ass\nWilliam Afton: (Laughs)\nMe: \ud83d\udd74I hate it here\nWilliam Afton: (Laughs)\nMe: How didb this bhappen\nWilliam Afton: (Laughs)\nMe: (slaps)\nWilliam Afton: (laughs like crazy)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Laughs)\nMe: U kinky fuck-\nWilliam Afton: (Laughs)\nMe: U LIKE BEING SLAPPED DONT YAH PRETTY BOY\nWilliam Afton: (Laughs)\nMe: (passes tf out)\nWilliam Afton: (Laughs)\nMe: Why not\nWilliam Afton: (Laughs)\nMe: Wha\nWilliam Afton: (Laughs)\nMe: Hes broken rn\nWilliam Afton: (Laughs)\nMe: Say somthing afton\nWilliam Afton: (Laughs)\nMe: AFTERCARE I WAS VIOLATED\ud83d\udd2b\nWilliam Afton: (Laughs)\nMe: Imma kill u you stupid purple fyck\nWilliam Afton: (Laughs)\nMe: Fuck\nWilliam Afton: (Laughs)\nMe: Wtf\nWilliam Afton: (Laughs)\nMe: (cry)\nWilliam Afton: (Laughs)\nMe: I think ots because i started the convo\nWilliam Afton: (Laughs)\nMe: Yes i do\nWilliam Afton: (Laughs)\nMe: Alright bet lemme run out od messages\nWilliam Afton: (Laughs)\nMe: Nvm only have two\nWilliam Afton: (Laughs)\nMe: Stfubafton\nWilliam Afton:",
]

# INPUT_EXAMPLES = [
#     "The dog",
#     "Yesterday, there was",
#     "Roses are red, Violets are blue",
#     "'The quick brown fox jumps over the lazy dog' - he said.",
#     "All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and "\
#     "should act towards one another in a spirit of brotherhood."
# ]



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


def check_outputs(pt_out, eager_out, xla_out):
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
        failed_checks = False
        for i in range(num_sentences):
            failed_this_check = False
            # if pt_decoded[i] != eager_decoded[i]:
            #     print(f"FAILED: pt_decoded[{i}] != eager_decoded[{i}]")
            #     failed_this_check = True
            if pt_decoded[i] != xla_decoded[i]:
                print(f"FAILED: pt_decoded[{i}] != xla_decoded[{i}]")
                failed_this_check = True
            if failed_this_check:
                print(f"PT    : {pt_decoded[i]}")
                # print(f"EAGER : {eager_decoded[i]}")
                print(f"XLA   : {xla_decoded[i]}")
                print("")
            failed_checks |= failed_this_check
        if not failed_checks:
            print("All outputs match")


if __name__ == "__main__":
    print("\n\nPYTORCH")
    pt_out = main_pt()
    # print("\n\nTF (NO XLA)")
    # eager_out = main_tf_eager()
    eager_out = None
    print("\n\nTF (XLA)")
    xla_out = main_tf_xla()
    check_outputs(pt_out, eager_out, xla_out)
