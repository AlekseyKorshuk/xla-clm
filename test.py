# Stand-alone TF XLA generate example for Decoder-Only Models.

# Note: execution times are deeply dependent on hardware.
# If you have a machine with a powerful GPU, I highly recommend you to try this example there!
import time
# import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM, AutoModelForCausalLM

# 1. Load model and tokenizer
# model_name = "gpt2"  # hakurei/litv2-6B-rev2
# remember: decoder-only models need left-padding
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-66b', padding_side="left", pad_token="</s>")

print(tokenizer.decode([85, 18, 275, 114, 52, 95, 1095, 964, 734, 50118,
                  5096, 35, 7586, 50118, 12645, 16384, 1438, 2871, 4832, 38,
                  581, 460, 657, 47, 53, 38, 240, 7, 517, 15,
                  734, 50118, 5096, 35, 7586, 50118, 12645, 16384, 1438, 2871,
                  4832, 38, 437, 6661, 734, 50118, 5096, 35, 7586, 50118,
                  12645, 16384, 1438, 2871, 4832, 40142, 734, 50118, 5096, 35,
                  7586, 50118, 12645, 16384]))
