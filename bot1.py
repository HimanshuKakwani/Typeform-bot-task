from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pandas as pd
import csv

# model_name = "microsoft/DialoGPT-large"
model_name = "microsoft/DialoGPT-medium"
# model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Welcome to Blabber-Bot. Let me get you started, please help me with your name")
lname = []
lnum = []
lemail = []
ledu = []

for step in range(5):
    text = input(">> User:")
    text = text.lower()
    if any(i in text for i in ["thank","thanks"]):
        res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","mention not"])
        print("Blabber-bot: ", res)
    elif any(i in text for i in ["himanshu", 'sakshi', "akash", "dev", 'jay','divya','meena','jack','john']):
        res = np.random.choice(["Could you please help me with your number (with country code).", "Please provide your number (with country code)"])
        print("Blabber-bot: ", res)
        lname.append(text)
    elif any (i in text for i in ["+91"]):
        res = np.random.choice(["Please provide your email id", "Please help me with your mail id"])
        print("Blabber-bot: ", res)
        lnum.append(text)
    elif any(i in text for i in ["@gmail",".com"]):
        res = np.random.choice(["Thank you, as a next step enter your educational qualifications", "Please enter your educational qualifications"])
        print("Blabber-bot: ", res)
        lemail.append(text)
    elif any(i in text for i in ["c.a.", "btech","college","school","mba","tenth","12th","engineer"]):
        res = np.random.choice(["Thank you very much, any other questions you would like me to answer(yes/no)?", "Thank you very much, any other questions you would like me to answer(yes/no)?"])
        print("Blabber-bot: ", res)
        ledu.append(text)
    elif any(i in text for i in ['yes', 'yeah']):
        res = np.random.choice(["Feel free to ask", "please go ahead"])
    elif any(i in text for i in ["exit","close"]):
        res = np.random.choice(["Tata","Have a good day","Bye","Goodbye","Hope to meet soon","peace out!"])
        print("Blabber-bot: ", res)

    
    else:
        # encode the input and add end of string token
        input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        chat_history_ids = model.generate(
            input_ids,
            max_length=1000,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            temperature=0.75,
            pad_token_id=tokenizer.eos_token_id
        )
        #print the output
        output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Blabber-Bot: {output}")

dict1 = {'name': lname, 'number':lnum, 'email': lemail, 'edu':ledu}
df = pd.DataFrame(dict1)
df.to_csv('trial.csv')
# print(lname)
# print(lnum)
# print(lemail)
# print(ledu)