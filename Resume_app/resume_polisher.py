from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr


api_key = "IYXxmg5tr-G4r-XWKOvsZqjRIToEgnS4CCkSpV5QQmsv"
project_id = "f8a9c89e-6333-4199-869a-8f4a5850b110"
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

def polish_resume(position_name, resume_content, polish_prompt=""):
    
    # Check if polish_prompt is provided
    if polish_prompt and polish_prompt.strip():
        prompt_use = f"Given the resume content: '{resume_content}', polish it based on the following instructions: {polish_prompt} for the {position_name} position."
    else:
        prompt_use = f"Suggest improvements for the following resume content: '{resume_content}' to better align with the requirements and expectations of a {position_name} position. Return the polished version, highlighting necessary adjustments for clarity, relevance, and impact in relation to the targeted role."


    messages = [
    {"role": "user",
     "content": [{"type": "text",
                "text": prompt_use
            },
        ]
    }
]

    generated_response = model.chat(messages=messages)
    generated_text = generated_response['choices'][0]['message']['content']

    return generated_text


resume_polish_application = gr.Interface(
    fn = polish_resume,
    flagging_mode = 'never',
    inputs = [
        gr.Textbox(label= 'Position Name', placeholder = "Enter the name of the position.."),
        gr.File(label = "Resume Content", file_types = [".pdf", ".docx", ".txt"]),
        gr.Textbox(label = "Polish Instruction (Optional)", placeholder="Enter specific instructions or areas for improvement (optional)...", lines=2),
    ],
    outputs = gr.Textbox(label = 'Polished Content', lines = 60),
    title = "Izzy's Resume Polish Application",
    description = "This application helps you polish your resume. Enter the position your want to apply, your resume content, and specific instructions or areas for improvement (optional), then get a polished version of your content."
)

resume_polish_application.launch()