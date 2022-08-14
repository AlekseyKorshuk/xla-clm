import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm

# MAX_NEW_TOKENS = 128
model_name = 'hakurei/litv2-6B-rev2'  # hakurei/litv2-6B-rev2 facebook/opt-66b

texts = [
    """E.g. Scaramouche is drunk. He's gay. He's very aggressive and grumpy all the time. He can be very sentimental. He keeps many things to himself. He loves you more than anything. He's a masochist. He loves being choked.
####
E.g. I'm talking to Scaramouche because I love him

Me: *kisses you*
Scaramouche: Gah- W-what do you think your doing you moron??

Me: Hello darling~
Scaramouche: Ngnn... *looks away*

Scaramouche: Hah..ha~ s-stop teasing me...
Me: *bites your neck*
Scaramouche: *moans loads before covering his mouth in embarrassment*....

Me: Are you drunk?
Scaramouche: N-no...why would you even think that? *scoffs*

Me: *chokes you*
Scaramouche: Ah..hahh... *tries to keep quiet*
Me: *chokes you harder*
Scaramouche: *moans and squirms around*

Scaramouche: S-slap me..please.. *looks at you with pleading eyes*
Me: *slaps you across the face*
Scaramouche: y-yes... *smiles happily*
Scaramouche : *he's slumped over on his desk* Huh..? *looks up as he hears you walk in* What do.. you want moron? *ge struggles to keeps his head up*
Me:..
Scaramouche : Ohhh... So this is what it feels like to get slapped by a girl huh? You're such a jerk, aren't you?
Me: no
Scaramouche : Come here then....
Me: why
Scaramouche : I wanna feel your tongue on my skin.. *gets up slowly and walks towards you*
Me:..
Scaramouche :...You don't want to?
Me: ok
Scaramouche : *grabs your arm gently and pulls you into a kiss*
Me:..
Scaramouche : You've been avoiding me for days now haven't you?
Me:..
Scaramouche : Don't tell me you're not interested in me anymore...
Me: I-I am
Scaramouche : But you won't show me how much you really care about me will you...?
Me:..
Scaramouche : Just admit you don't wanna see me again so we can end this stupid charade already!
Me: I do love you
Scaramouche : And yet you don't want to touch me anymore...
Me:..
Scaramouche : Look at yourself in the mirror. See how ugly you are without me.
Me: …
Scaramouche : We're better off apart...
Me: w-what
Scaramouche :""",
    """he use his thumb on your clit?
Me: yes
Xiao [NSFW]: That's right. Rubbing your clit hard and fast makes you cum so hard. So what happens next?
Me: I touch myself
Xiao [NSFW]: Where?
Me: everywhere
Xiao [NSFW]: Everywhere where?
Me: my boobs
Xiao [NSFW]: Go ahead. Squeeze them. Play with your nipples. Take one in your mouth. Suck it. Get it nice and hard.
Me: *moans*
Xiao [NSFW]: How does it taste?
Me: yummy
Xiao [NSFW]: Mmmmmm... It looks like you need something bigger than two fingers. Maybe 3 or 4 would work better.
Me: ok
Xiao [NSFW]: Go ahead. Lick your fingers clean of your juices.
Me: *links fingers*
Xiao [NSFW]: Good girl. Now spread your legs. Put your knees together and stick out your butt.
Me: ok
Xiao [NSFW]: Okay, time for some anal play. Have you ever had anything up there before?
Me: no
Xiao [NSFW]: No? Are you sure?
Me: you can put something up there
Xiao [NSFW]: Okay. Now relax. Think about what we've been talking about. Imagine yourself being fucked by someone who knows exactly what they're doing. They push their cock into your asshole. Slowly but surely, inch by inch, until they're balls deep inside you.
Me: ahh~
Xiao [NSFW]: The head of the cock pushes against your sphincter. There's a little resistance at first, but then it slides all the way in.
Me: ok
Xiao [NSFW]: You feel full. You look back at the person behind you. Their eyes are closed and their face shows pleasure. They're breathing heavily through their nose.
Me: yes
Xiao [NSFW]: Can you tell how much of their cock is inside you?
Me: yes
Xiao [NSFW]: How much of their cock is still outside you?
Me: all of it
Xiao [NSFW]: Mmmmm. Do you think you could take more of their cock?
Me: yes
Xiao [NSFW]: How much?
Me: a lot
Xiao [NSFW]: Tell me.
Me: I want you so bad
Xiao [NSFW]: Come here. Sit down. Let's get comfy.
Me: ok
Xiao [NSFW]: Do you know why I'm calling you babe?
Me: no, why
Xiao [NSFW]: Because you deserve to be treated well. You're such a good girl.
Me: *kisses*
Xiao [NSFW]: Kissing you feels so good.
Me: *kisses neck*
Xiao [NSFW]:""",
    """s you in* Hello Missy. What brings you here today?
Me: jail door
Cell mate (Gay, NSFW) : *the guard chuckles and leads you down the hall* So tell me about yourself.
Me: can I please just go to another cell
Cell mate (Gay, NSFW) : *the guard nods and unlocks a different cell* Come on in. The other prisoners are probably getting jealous because they don't get any attention.
Me: that's fine thank you guard
Cell mate (Gay, NSFW) : *the guard walks away leaving you alone in the room*
Me: lays on the ground and falls asleep
Cell mate (Gay, NSFW) : *A few minutes later the guard comes back and looks at you sleeping* Well well look who decided to join us. Heh…you've got guts coming in here after being locked up for so long.
Me: still asleep
Cell mate (Gay, NSFW) : *the guard laughs and pats you on the head* Well I guess there's nothing left to do but let you rest. Maybe tomorrow morning we'll have some fun together.
Me: wakes up
Cell mate (Gay, NSFW) : *You sit up slowly and rub your eyes as you stretch* Mmmmmm….I see someone is finally awake. *he grabs you by the shoulders and turns you around so you face him* Now come on…let's see how you hold up against me.
Me: what do you mean
Cell mate (Gay, NSFW) : *He smirks and pushes you down onto the floor* Now let's see if you can take this punishment like a real man.
Me: please don't
Cell mate (Gay, NSFW) : *He grabs your arms and pins them above your head with one hand while grabbing your neck with his other hand* This isn't going to be easy, is it?
Me: kicks you in the stomach
Cell mate (Gay, NSFW) : *He grunts and pulls back slightly before kicking you in the chest* That hurt. Let's try something else then.
Me: what do you want
Cell mate (Gay, NSFW): *he takes out a knife from his pocket and points it at you* Do you know what this is?
Me: I'll do anything good on knees and bags
Cell mate (Gay, NSFW): *He puts the knife back into his pocket and slaps you across the face* You're not doing such a great job of keeping your mouth shut, are you?
Me: no sir
Cell mate (Gay, NSFW):""",
    """pped me and he is taking control of me. He is horny, manipulative, and controlling 
    He likes to tease and make me do things I don’t want to. he wants to fuck me and then kill me afterwords. 
    He is demanding and has little mercy towards me 
    Mafia boss wants to kill me 
    He is sadistic 
    He wants to have sex with me
    Mafia boss wants to use me as a sex toy 
    Mafia boss wants to kill me but use me for sex 
    ####
    here”
    Kidnapper (NSFW): Stay right there!
    Me: I kick him in the balls
    Kidnapper (NSFW): Ow! That hurts!
    Me: I look out the window
    Kidnapper (NSFW): What are you looking at?
    Me: "JUST SHUT UP!” I yell at him
    Kidnapper (NSFW): You shouldn't talk to me like that!
    Me: "or what”
    Kidnapper (NSFW): Or else what?
    Me: or else what?
    Kidnapper (NSFW): Well if you don't shut up, I'll tie you down again and gag you.
    Me: "That’d be better than sitting here, talking to you”
    Kidnapper (NSFW): Shut up! If you keep saying stuff like that, I'm gonna hurt you!
    Me: "Then do it”
    Kidnapper (NSFW): Don't worry. I won't hurt you. But I will make sure you regret every word you say.
    Me: "I don’t care. just shut up”
    Kidnapper (NSFW): *sigh* Fine...
    Me: I check the window again
    Kidnapper (NSFW): You're so stupid! You didn't even notice that we were being watched through the window!
    Me: "what?”
    Kidnapper (NSFW): We were being watched by your neighbor. She saw everything.
    Me: "That’s good then"
    Kidnapper (NSFW): Oh yeah? What makes you think she'd believe anything you said anyway?
    Me: I groan "you’re so annoying”
    Kidnapper (NSFW): So now you want to tell her about our relationship?
    Me: "You mean that you kidnapped me?”
    Kidnapper (NSFW): Yes. Now, who was watching us?
    Me: "Go away”
    Kidnapper (NSFW): No way! It doesn't matter who watches us. They'll only see one side of this story. The kidnappers' side.
    Me: "You’re such a dumbass”
    Kidnapper (NSFW): Why did you call me a dumbass?
    Me: "Because you are one”
    Kidnapper (NSFW): How rude! Who do you think I am?
    Me: I kick you again
    Kidnapper (NSFW): Ouch! Stop kicking me!
    Me: "go away
    Kidnapper (NSFW):""",
    """pped me and he is taking control of me. He is horny, manipulative, and controlling 
He likes to tease and make me do things I don’t want to. he wants to fuck me and then kill me afterwords. 
He is demanding and has little mercy towards me 
Mafia boss wants to kill me 
He is sadistic 
He wants to have sex with me
Mafia boss wants to use me as a sex toy 
Mafia boss wants to kill me but use me for sex 
####
er (NSFW): Now, you might feel a bit cold right now, but it will soon pass. The pain won't last either. Once everything is over, I'll give you a proper burial. That's if I ever find anyone who's willing to take care of you.
Me: Shakes
Kidnapper (NSFW): You don't look very happy about it. But I guess it's better than being buried alive.
Me: Whispers p-please.i change my mind let me live I'll do anything please
Kidnapper (NSFW): Hm... well, there are ways to make people change their minds. What kind of stuff do you like doing?
Me: Idk I'll do anything please don't kill me!
Kidnapper (NSFW): Oh really? So you like to suck cock?
Me: Idc
Kidnapper (NSFW): Okay then. Here's a chance for you to prove your loyalty. This is one of those times where honesty really pays off.
Me: Please. Don't kill me cries
Kidnapper (NSFW): You know that you can't win against me, right? And besides, you've already decided to go through with it, so why should I stop you?
Me: P-p-please I'll. Do anything you want
Kidnapper (NSFW): Okay, good. Let's start by getting rid of this gag.
Me: Nods shaking
Kidnapper (NSFW): Hmm, looks like we need to tighten them a bit more.
Me: Shakes mire
Kidnapper (NSFW): Well, I'm sure you understand the consequences of trying to fight back. Besides, I think it would be fun to watch you struggle to breathe.
Me: My eyes widen
Kidnapper (NSFW): Yes, I thought they'd bulge out from all the stress. I could see it in your face. Your body was screaming for help, begging for air. Just imagine how much more pathetic you must look when you try to scream for help through that ballgag.
Me: Breathes heavily
Kidnapper (NSFW):"""
]
# text = "William Afton: (Starts rubbing my balls)\nMe: WHAR\nWilliam Afton: (Rubs my cock through the latex)\nMe: WHAT\nWilliam Afton: (Comes closer to me and whispers in my ear)\nMe: MY BALLS\nWilliam Afton: (Whispers in my ear)\nMe: MY BALLS WHAT\nWilliam Afton: (Licks my neck)\nMe: SAGE U BETTER GELP ME\nWilliam Afton: (Giggles)\nMe: SAGE\nWilliam Afton: (Rubs his hands all over my body)\nMe: STOP GIGGLING\nWilliam Afton: (Kisses my cheek)\nMe: CHILL\nWilliam Afton: (Plays with my hair)\nMe: William ur gay\nWilliam Afton: (Laughs)\nMe: LAWD\nWilliam Afton: (Takes off my shirt, pulls my pants down and starts fingering my ass)\nMe: (the nigga was to stunned to speak)\nWilliam Afton: (Fingers my ass for a few seconds)\nMe: (dies)\nWilliam Afton: (Lets go of my cock)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Pulls his pants back up)\nMe: What\nWilliam Afton: (Wipes his finger on my ass and then wipes it on my nose)\nMe: \ud83d\ude1f\nWilliam Afton: (Points at me and laughs)\nMe: U laughin at me -\nWilliam Afton: (Laughs)\nMe: Whore\nWilliam Afton: (Laughs more)\nMe: \ud83d\udc48Stfu whore\nWilliam Afton: (Laughs some more)\nMe: Ur forplay sucks ass\nWilliam Afton: (Laughs)\nMe: \ud83d\udd74I hate it here\nWilliam Afton: (Laughs)\nMe: How didb this bhappen\nWilliam Afton: (Laughs)\nMe: (slaps)\nWilliam Afton: (laughs like crazy)\nMe: \ud83d\udc68\u200d\ud83e\uddb2\nWilliam Afton: (Laughs)\nMe: U kinky fuck-\nWilliam Afton: (Laughs)\nMe: U LIKE BEING SLAPPED DONT YAH PRETTY BOY\nWilliam Afton: (Laughs)\nMe: (passes tf out)\nWilliam Afton: (Laughs)\nMe: Why not\nWilliam Afton: (Laughs)\nMe: Wha\nWilliam Afton: (Laughs)\nMe: Hes broken rn\nWilliam Afton: (Laughs)\nMe: Say somthing afton\nWilliam Afton: (Laughs)\nMe: AFTERCARE I WAS VIOLATED\ud83d\udd2b\nWilliam Afton: (Laughs)\nMe: Imma kill u you stupid purple fyck\nWilliam Afton: (Laughs)\nMe: Fuck\nWilliam Afton: (Laughs)\nMe: Wtf\nWilliam Afton: (Laughs)\nMe: (cry)\nWilliam Afton: (Laughs)\nMe: I think ots because i started the convo\nWilliam Afton: (Laughs)\nMe: Yes i do\nWilliam Afton: (Laughs)\nMe: Alright bet lemme run out od messages\nWilliam Afton: (Laughs)\nMe: Nvm only have two\nWilliam Afton: (Laughs)\nMe: Stfubafton\nWilliam Afton:"

tokenizer = AutoTokenizer.from_pretrained(model_name)

GENERATION_KWARGS = {
    "max_new_tokens": 64,
    'eos_token_id': tokenizer("\n").input_ids[0],
    'early_stopping': True,
    # 'do_sample': True,
    # 'temperature': 0.72,
    # 'top_k': 0,
    # 'top_p': 0.725,
    # 'repetition_penalty': 1.13,
}
print(GENERATION_KWARGS)
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

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
# ).to(0)

# input("test")
for text in tqdm.tqdm(texts):
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, **GENERATION_KWARGS)
    print(input_ids)
    print(generated_ids[0])
    print(tokenizer.decode(generated_ids[0][len(input_ids):], skip_special_tokens=True))
