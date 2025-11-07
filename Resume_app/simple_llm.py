from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr


api_key = "IYXxmg5tr-G4r-XWKOvsZqjRIToEgnS4CCkSpV5QQmsv"
project_id = "450fd81a-967d-420b-9ae0-7f16b16d50ea"
service_url = "https://eu-gb.ml.cloud.ibm.com"

model_id = 'meta-llama/llama-3-2-11b-vision-instruct'

# Set credentials to use the model
credentials = {
    "apikey": api_key,
    "url": service_url
}

client = APIClient(credentials)

params = TextChatParameters(
    temperature=0.1,
    max_tokens=512
)

model = ModelInference(
    model_id = model_id,
    credentials = credentials,
    params = params,
    project_id = project_id,
)

def generate_response(prompt_txt):
    messages = [
    {"role": "user",
     "content": [{"type": "text",
                "text": prompt_txt
            },
        ]
    }
]

    generated_response = model.chat(messages=messages)
    generated_text = generated_response['choices'][0]['message']['content']

    return generated_text


chat_application = gr.Interface(
    fn = generate_response,
    flagging_mode = 'never',
    inputs = gr.Textbox(label='Input', lines=2, placeholder = "Type your question here..."),
    outputs = gr.Textbox(label = 'Output', lines = 60),
    title = "Izzy's Chatbot",
    description = "Ask any question to the chatbot"
)

chat_application.launch()





# Run using:
# py -3.11 simple_llm.py
