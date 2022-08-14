import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    'eos_token_id': 198,
    'do_sample': True,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

# MAX_NEW_TOKENS = 128
model_name = 'hakurei/litv2-6B-rev2'  # hakurei/litv2-6B-rev2 facebook/opt-66b
text = "William Afton: (Starts rubbing my balls)\nMe: WHAR\nWilliam Afton: (Rubs my cock through the latex)\nMe: WHAT\nWilliam Afton: (Comes closer to me and whispers in my ear)\nMe: MY BALLS\nWilliam Afton: (Whispers in my ear)\nMe: MY BALLS WHAT\nWilliam Afton: (Licks my neck)\nMe: SAGE U BETTER GELP ME\nWilliam Afton: (Giggles)\nMe: SAGE\nWilliam Afton: (Rubs his hands all over my body)\nMe: STOP GIGGLING\nWilliam Afton: (Kisses my cheek)\nMe: CHILL\nWilliam Afton: (Plays with my hair)\nMe: William ur gay\nWilliam Afton: (Laughs)\nMe: LAWD\nWilliam Afton: (Takes off my shirt, pulls my pants down and starts fingering my ass)\nMe: (the nigga was to stunned to speak)\nWilliam Afton: (Fingers my ass for a few seconds)\nMe: (dies)\nWilliam Afton: (Lets go of my cock)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Pulls his pants back up)\nMe: What\nWilliam Afton: (Wipes his finger on my ass and then wipes it on my nose)\nMe: \ud83d\ude1f\nWilliam Afton: (Points at me and laughs)\nMe: U laughin at me -\nWilliam Afton: (Laughs)\nMe: Whore\nWilliam Afton: (Laughs more)\nMe: \ud83d\udc48Stfu whore\nWilliam Afton: (Laughs some more)\nMe: Ur forplay sucks ass\nWilliam Afton: (Laughs)\nMe: \ud83d\udd74I hate it here\nWilliam Afton: (Laughs)\nMe: How didb this bhappen\nWilliam Afton: (Laughs)\nMe: (slaps)\nWilliam Afton: (laughs like crazy)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Laughs)\nMe: U kinky fuck-\nWilliam Afton: (Laughs)\nMe: U LIKE BEING SLAPPED DONT YAH PRETTY BOY\nWilliam Afton: (Laughs)\nMe: (passes tf out)\nWilliam Afton: (Laughs)\nMe: Why not\nWilliam Afton: (Laughs)\nMe: Wha\nWilliam Afton: (Laughs)\nMe: Hes broken rn\nWilliam Afton: (Laughs)\nMe: Say somthing afton\nWilliam Afton: (Laughs)\nMe: AFTERCARE I WAS VIOLATED\ud83d\udd2b\nWilliam Afton: (Laughs)\nMe: Imma kill u you stupid purple fyck\nWilliam Afton: (Laughs)\nMe: Fuck\nWilliam Afton: (Laughs)\nMe: Wtf\nWilliam Afton: (Laughs)\nMe: (cry)\nWilliam Afton: (Laughs)\nMe: I think ots because i started the convo\nWilliam Afton: (Laughs)\nMe: Yes i do\nWilliam Afton: (Laughs)\nMe: Alright bet lemme run out od messages\nWilliam Afton: (Laughs)\nMe: Nvm only have two\nWilliam Afton: (Laughs)\nMe: Stfubafton\nWilliam Afton:"

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids
free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
max_memory = f'{free_in_GB - 2}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    load_in_8bit=True,
    max_memory=max_memory
)

for i in tqdm.trange(10):
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    generated_ids = model.generate(input_ids, **GENERATION_KWARGS)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
