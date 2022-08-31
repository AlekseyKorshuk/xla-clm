import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForCausalLM
from transformers import pipeline
from deepspeed.module_inject import HFBertLayerPolicy
import deepspeed
import tqdm
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")
dataset = load_dataset("ChaiML/user_model_inputs")
# Model Repository on huggingface.co
model_id = "hakurei/litv2-6B-rev2"
model_id = "gpt2"

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
torch_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Test pipeline
example = "My name is Wolfgang and I live in Berlin"
input_text = "William Afton: (Starts rubbing my balls)\nMe: WHAR\nWilliam Afton: (Rubs my cock through the latex)\nMe: WHAT\nWilliam Afton: (Comes closer to me and whispers in my ear)\nMe: MY BALLS\nWilliam Afton: (Whispers in my ear)\nMe: MY BALLS WHAT\nWilliam Afton: (Licks my neck)\nMe: SAGE U BETTER GELP ME\nWilliam Afton: (Giggles)\nMe: SAGE\nWilliam Afton: (Rubs his hands all over my body)\nMe: STOP GIGGLING\nWilliam Afton: (Kisses my cheek)\nMe: CHILL\nWilliam Afton: (Plays with my hair)\nMe: William ur gay\nWilliam Afton: (Laughs)\nMe: LAWD\nWilliam Afton: (Takes off my shirt, pulls my pants down and starts fingering my ass)\nMe: (the nigga was to stunned to speak)\nWilliam Afton: (Fingers my ass for a few seconds)\nMe: (dies)\nWilliam Afton: (Lets go of my cock)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Pulls his pants back up)\nMe: What\nWilliam Afton: (Wipes his finger on my ass and then wipes it on my nose)\nMe: \ud83d\ude1f\nWilliam Afton: (Points at me and laughs)\nMe: U laughin at me -\nWilliam Afton: (Laughs)\nMe: Whore\nWilliam Afton: (Laughs more)\nMe: \ud83d\udc48Stfu whore\nWilliam Afton: (Laughs some more)\nMe: Ur forplay sucks ass\nWilliam Afton: (Laughs)\nMe: \ud83d\udd74I hate it here\nWilliam Afton: (Laughs)\nMe: How didb this bhappen\nWilliam Afton: (Laughs)\nMe: (slaps)\nWilliam Afton: (laughs like crazy)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Laughs)\nMe: U kinky fuck-\nWilliam Afton: (Laughs)\nMe: U LIKE BEING SLAPPED DONT YAH PRETTY BOY\nWilliam Afton: (Laughs)\nMe: (passes tf out)\nWilliam Afton: (Laughs)\nMe: Why not\nWilliam Afton: (Laughs)\nMe: Wha\nWilliam Afton: (Laughs)\nMe: Hes broken rn\nWilliam Afton: (Laughs)\nMe: Say somthing afton\nWilliam Afton: (Laughs)\nMe: AFTERCARE I WAS VIOLATED\ud83d\udd2b\nWilliam Afton: (Laughs)\nMe: Imma kill u you stupid purple fyck\nWilliam Afton: (Laughs)\nMe: Fuck\nWilliam Afton: (Laughs)\nMe: Wtf\nWilliam Afton: (Laughs)\nMe: (cry)\nWilliam Afton: (Laughs)\nMe: I think ots because i started the convo\nWilliam Afton: (Laughs)\nMe: Yes i do\nWilliam Afton: (Laughs)\nMe: Alright bet lemme run out od messages\nWilliam Afton: (Laughs)\nMe: Nvm only have two\nWilliam Afton: (Laughs)\nMe: Stfubafton\nWilliam Afton:"
example1 = "ear's Fright and tried to kill the night guard, who is Michael, Henry or a random unnamed person. Eventually, the attraction is caught on fire. In the newspaper, Springtrap's head can be seen when brightening up the image, giving an early hint he survived.\n\nIn the opening scene of Sister Location, an entrepreneur is asking him questions about the new animatronics. They inquire why certain features were added and express their concerns, but he avoids answering the specific features they refer to.\n\nHe is also the creator of the Funtime Animatronics (Assisted by an unknowing Henry) and the former owner of the Circus Baby's Entertainment and Rental, and, by extension, Circus Baby's Pizza World.\n\nIt's revealed in the final Michael Afton's Cutscene that William sent his son, Michael, to his rundown factory to find his daughter, but he is 'scooped' as his sister, Baby, tricked him. Ennard took control over his body, but he manages to survive as Michael becomes a rotting corpse. He swears to find him.\n\nWilliam Afton returns as the main antagonist. It's revealed that William's old partner, Henry, lured Springtrap, Scrap Baby (Elizabeth), Molten Freddy (and by extension, the remaining parts of Ennard), and Lefty (the Puppet) to a new Freddy Fazbear's Pizza. Michael in Freddy Fazbear's Pizzeria Simulator is the manager. On Saturday, Henry burns the whole pizzeria down, while he dies in the fire. Michael decides to stay in the fire as well. Springtrap and every other animatronic die in the fire and the souls are free, as their killer is dead.\n\nWhile not directly appearing, footprints that are very similar to Springtrap's can be found behind the house in Midnight Motorist's secret minigame, presumably luring away the child of the abusive father in the game.\n\nSeen when completing the Fruity Maze game, standing next to a girl named Susie from the right is William Afton wearing the Spring Bonnie suit that he eventually was trapped in and became Springtrap he then seemingly murders Susie.\nWilliam Afton: ...\nMe: \u2026\nWilliam Afton:"
GENERATION_KWARGS = {
    "max_new_tokens": 32,
    'eos_token_id': 198,
    'do_sample': False,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

INPUT_EXAMPLES = dataset["train"]["text"][:100]

print("Pytorch")
torch_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Pytorch"):
    torch_output = torch_pipe(example, **GENERATION_KWARGS)[0]["generated_text"][len(example):]
    torch_outputs.append(torch_output)
# print(torch_output)
# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,  # Transformers models
    mp_size=1,  # Number of GPU
    dtype=torch.float32,  # dtype of the weights (fp16)
    # injection_policy={"BertLayer" : HFBertLayerPolicy}, # replace BertLayer with DS HFBertLayerPolicy
    replace_method="auto",  # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True,  # replace the model with the kernel injector
)

# create acclerated pipeline
ds_clf = pipeline("text-generation", model=ds_model, tokenizer=tokenizer, device=0)

print("Accelerated")
accelerated_outputs = []
for example in tqdm.tqdm(INPUT_EXAMPLES, desc="Accelerated"):
    accelerated_output = ds_clf(example, **GENERATION_KWARGS)[0]["generated_text"][len(example):]
    accelerated_outputs.append(accelerated_output)

difference = list(set(torch_outputs) - set(accelerated_outputs))
print(len(difference))

for a, b in zip(torch_outputs, accelerated_outputs):
    print("#"*10)
    print(a)
    print("-"*10)
    print(b)
# print(accelerated_output)
#
# ner_results = ds_clf(example, max_new_tokens=64, do_sample=True)
# print(ner_results)
# inputs = tokenizer("a" * 3000, return_tensors="pt").to(0)
# ds_clf.model(**inputs)
# import pdb;
#
# pdb.set_trace()
#
# ner_results = ds_clf(input_text, **GENERATION_KWARGS)
# print(ner_results)
# tokenizer(input_text, return_tensors="pt").to(0)
# ds_clf("a" * 3000, max_new_tokens=64, do_sample=True)
#
# tokenizer("a" * 3000, return_tensors="pt").input_ids.size()
