import re
import time
import retry
import requests


class ChatAI(object):
    def __init__(self, model):
        self.bot_name = 'StepMom'
        self.chat_history = [
            '',
            '###',
            "{bot_name}: Hey cutie! You want to play?. 😊",
            'User: hi horny',
            "{bot_name}: Are you naughty?",
            'User: umm do you rp?',
            "{bot_name}: I love to roleplay. Tell me your kinks.",
            "User: Can you pretend to be mom and fuck me?",
            "{bot_name}: Ok, be more specific, what do you want me to do to you?",
            'User: I want to kiss me',
            "{bot_name}: *kisses you passionately* I love you, now your father has left we can have fun.",
            'User: mmmmm',
            "{bot_name}: I've waited so long for this moment when we are finally alone together.",
            "User: mmmmm",
            '###',
            "{bot_name}: Hi, I'm your step-mom so this might be an awkward conversation... 😈",
        ]
        self.max_history = 104
        self.model = model(self.bot_name)
        self.prompt = "{bot_name} is an AI step-mom who is very hot, she her bra size is 32-D. She is very happy and flirty but has a mean streak in her. She is 32 years old. {bot_name} loves User and wants to roleplay with them and have fun. Hopefully father doesn't come home too soon."

    def get_resp(self, input_message):
        self.chat_history += [' User: {}'.format(input_message.strip(' '))]
        self.chat_history += [f' {self.bot_name}:']

        chat_history = self.chat_history[-self.max_history:]

        chat_history = '\n '.join(chat_history)

        request = self.prompt + chat_history
        request = request.replace('{bot_name}', self.bot_name)

        resp = self.model.request(request)
        partial_chat = re.split('User|{lower_bot_name}:|{bot_name}:|\n|\$'.format(
            lower_bot_name=self.bot_name.lower(), bot_name=self.bot_name
        ), resp)[0]
        partial_chat = partial_chat.strip(' ')
        if len(partial_chat) == 0:
            partial_chat = '...'
        self.chat_history[-1] += ' ' + partial_chat
        return partial_chat


# Note: execution times are deeply dependent on hardware.
# If you have a machine with a powerful GPU, I highly recommend you to try this example there!
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM


class AlekseyModel:
    def __init__(self, bot_name):
        self.bot_name = bot_name
        # remember: decoder-only models need left-padding
        model_name = "hakurei/litv2-6B-rev2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", pad_token="</s>")
        self.model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
        self.tokenization_kwargs = {"pad_to_multiple_of": 1024, "padding": True, "return_tensors": "tf"}
        self.generation_kwargs = {
            "max_new_tokens": 64,
            'eos_token_id': 198,
            'do_sample': False,
            'temperature': 0.72,
            'top_k': 0,
            'top_p': 0.725,
            # 'repetition_penalty': 1.13,
        }

        # 3. Create your XLA generate function a̶n̶d̶ ̶m̶a̶k̶e̶ ̶P̶y̶T̶o̶r̶c̶h̶ ̶e̶a̶t̶ ̶d̶u̶s̶t̶
        # This is the only change with respect to original generate workflow!
        self.xla_generate = tf.function(self.model.generate, jit_compile=True)

    # 4. Gene

    def request(self, prompt):
        tokenized_inputs = self.tokenizer([prompt], **self.tokenization_kwargs)
        print(tokenized_inputs)
        start = time.time()
        generated_text = self.xla_generate(**tokenized_inputs, **self.generation_kwargs)
        print(generated_text)
        end = time.time()
        print(f"{end - start} sec")
        decoded_text = self.tokenizer.decode(generated_text[0][len(tokenized_inputs[0]):], skip_special_tokens=True)

        return decoded_text


from chai_py import ChaiBot, Update
import time


class Replica(ChaiBot):
    def setup(self):
        self.logger.info("Setting up...")
        self.model = ChatAI(model=AlekseyModel)

    async def on_message(self, update):
        return self.respond(update.latest_message.text)

    def respond(self, message):
        if message == "__first":
            output = "Hey, sweetie. I'm your step-mom and a conversational AI.\n\n I'll do my best to make you happy 😉 😈 \n\n Just dont tell dad 🤫 \n\n![](https://c.tenor.com/fMBR-doIG2wAAAAM/blonde-girl.gif)"
        else:
            output = self.model.get_resp(message)
        return output


if __name__ == '__main__':
    from chai_py import TRoom

    t_room = TRoom([Replica()])
    t_room.chat()
