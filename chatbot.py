'''
We will use model_name = "facebook/blenderbot-400M-distill", basic version, not for complex operations
'''

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

conversation_history = []              # Create a list to save converstaion history

# During each interaction, you will pass your conversation history to the model along with your input 
# so that it may also reference the previous conversation when generating the next answer.

# The transformers library function expects to receive/process conversation history as tokens
# Each token seperated with '\n', thus use join() to create a string

history_string = '\n'.join(conversation_history)

text = 'Hello, how are you doing?'
inputs = tokenizer.encode_plus(history_string, text, return_tensors='pt')
# after this, run "print(inputs)", to generate a Python Dictionary that contain special keywords that allow the model to properly reference its contents.
# run it in terminal (python chatbot.py)

outputs = model.generate(**inputs)
# after this, run "print(outputs)", to see dictionary and tokens 

# since outputs is a dictionary with tokens and not words, we have to decode to see them in plaintext
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()      # strip() removes whitespace
# run print(response)
print(response)


# Update conversation history
conversation_history.append(text)
conversation_history.append(response)

# run python chatbot.py




