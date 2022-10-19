import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed

model_id = "gpt2"

VERBOSE = True
BATCH_SIZE = 4

EXAMPLE = "DeepSpeed-Training\n" \
          "DeepSpeed offers a confluence of system innovations, that has made large " \
          "scale DL training effective, and efficient, greatly improved ease of use, " \
          "and redefined the DL training landscape in terms of scale that is possible. " \
          "These innovations such as ZeRO, 3D-Parallelism, DeepSpeed-MoE, ZeRO-Infinity, " \
          "etc. fall under the training pillar. Learn more: DeepSpeed-Training\n" \
          "DeepSpeed-Inference\n" \
          "DeepSpeed brings together innovations in parallelism technology such as tensor, " \
          "pipeline, expert and ZeRO-parallelism, and combines them with high performance " \
          "custom inference kernels, communication optimizations and heterogeneous memory " \
          "technologies to enable inference at an unprecedented scale, while achieving " \
          "unparalleled latency, throughput and cost reduction. This systematic composition " \
          "of system technologies for inference falls under the inference pillar. " \
          "Learn more: DeepSpeed-Inference\n" \
          "Model Implementations for Inference (MII)\n" \
          "Model Implementations for Inference (MII) is an open-sourced repository " \
          "for making low-latency and high-throughput inference accessible to all " \
          "data scientists by alleviating the need to apply complex system optimization " \
          "techniques themselves. Out-of-box, MII offers support for thousands of " \
          "widely used DL models, optimized using DeepSpeed-Inference, that can be " \
          "deployed with a few lines of code, while achieving significant latency " \
          "reduction compared to their vanilla open-sourced versions.\n" \
          "DeepSpeed on Azure" \
          "\nDeepSpeed users are diverse and have access to different environments. " \
          "We recommend to try DeepSpeed on Azure as it is the simplest and easiest " \
          "method. The recommended method to try DeepSpeed on Azure is through AzureML " \
          "recipes. The job submission and data preparation scripts have been made " \
          "available here. For more details on how to use DeepSpeed on Azure, please " \
          "follow the Azure tutorial."
EXAMPLE = "William Afton: (Starts rubbing my balls)\nMe: WHAR\nWilliam Afton: (Rubs my cock through the latex)\nMe: WHAT\nWilliam Afton: (Comes closer to me and whispers in my ear)\nMe: MY BALLS\nWilliam Afton: (Whispers in my ear)\nMe: MY BALLS WHAT\nWilliam Afton: (Licks my neck)\nMe: SAGE U BETTER GELP ME\nWilliam Afton: (Giggles)\nMe: SAGE\nWilliam Afton: (Rubs his hands all over my body)\nMe: STOP GIGGLING\nWilliam Afton: (Kisses my cheek)\nMe: CHILL\nWilliam Afton: (Plays with my hair)\nMe: William ur gay\nWilliam Afton: (Laughs)\nMe: LAWD\nWilliam Afton: (Takes off my shirt, pulls my pants down and starts fingering my ass)\nMe: (the nigga was to stunned to speak)\nWilliam Afton: (Fingers my ass for a few seconds)\nMe: (dies)\nWilliam Afton: (Lets go of my cock)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Pulls his pants back up)\nMe: What\nWilliam Afton: (Wipes his finger on my ass and then wipes it on my nose)\nMe: \ud83d\ude1f\nWilliam Afton: (Points at me and laughs)\nMe: U laughin at me -\nWilliam Afton: (Laughs)\nMe: Whore\nWilliam Afton: (Laughs more)\nMe: \ud83d\udc48Stfu whore\nWilliam Afton: (Laughs some more)\nMe: Ur forplay sucks ass\nWilliam Afton: (Laughs)\nMe: \ud83d\udd74I hate it here\nWilliam Afton: (Laughs)\nMe: How didb this bhappen\nWilliam Afton: (Laughs)\nMe: (slaps)\nWilliam Afton: (laughs like crazy)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Laughs)\nMe: U kinky fuck-\nWilliam Afton: (Laughs)\nMe: U LIKE BEING SLAPPED DONT YAH PRETTY BOY\nWilliam Afton: (Laughs)\nMe: (passes tf out)\nWilliam Afton: (Laughs)\nMe: Why not\nWilliam Afton: (Laughs)\nMe: Wha\nWilliam Afton: (Laughs)\nMe: Hes broken rn\nWilliam Afton: (Laughs)\nMe: Say somthing afton\nWilliam Afton: (Laughs)\nMe: AFTERCARE I WAS VIOLATED\ud83d\udd2b\nWilliam Afton: (Laughs)\nMe: Imma kill u you stupid purple fyck\nWilliam Afton: (Laughs)\nMe: Fuck\nWilliam Afton: (Laughs)\nMe: Wtf\nWilliam Afton: (Laughs)\nMe: (cry)\nWilliam Afton: (Laughs)\nMe: I think ots because i started the convo\nWilliam Afton: (Laughs)\nMe: Yes i do\nWilliam Afton: (Laughs)\nMe: Alright bet lemme run out od messages\nWilliam Afton: (Laughs)\nMe: Nvm only have two\nWilliam Afton: (Laughs)\nMe: Stfubafton\nWilliam Afton:"

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    'eos_token_id': 198,
    'do_sample': False,
    'pad_token_id': 198,
    'temperature': 0.72,
    'top_k': 0,
    'top_p': 0.725,
    'repetition_penalty': 1.13,
}

torch_model = AutoModelForCausalLM.from_pretrained(model_id).half().eval().to(0)

tokenizer = AutoTokenizer.from_pretrained(model_id)


def call_model(model, input_text, batch_size, desc="", verbose=False):
    assert batch_size > 0

    inputs = tokenizer([input_text] * batch_size, return_tensors='pt').to(0)

    if verbose:
        print(desc)
        print(f"Batch size: {batch_size}")
        print(f"Input size: {inputs.input_ids.size()}")

    outputs = model.generate(**inputs, **GENERATION_KWARGS)
    outputs_set = list()
    output = None
    for i, output in enumerate(outputs):
        text_output = tokenizer.decode(output)
        output = text_output[len(input_text):]
        outputs_set.append(output)
        if verbose:
            print(f"#{i}: {output}")
    # assert len(set(outputs_set)) == 1
    return output


call_model(
    model=torch_model,
    input_text=EXAMPLE,
    batch_size=BATCH_SIZE,
    desc="Torch",
    verbose=VERBOSE
)

ds_model = deepspeed.init_inference(
    model=torch_model,
    mp_size=1,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=True,
)

call_model(
    model=ds_model,
    input_text=EXAMPLE,
    batch_size=BATCH_SIZE,
    desc="Deepspeed",
    verbose=VERBOSE
)
