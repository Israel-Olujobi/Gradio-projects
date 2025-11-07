from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr


api_key = "IYXxmg5tr-G4r-XWKOvsZqjRIToEgnS4CCkSpV5QQmsv"
project_id = "50718e93-32cb-45c1-a592-e5334440de06"
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

def generate_career_advice(position_applied, resume_content, job_description):
        
    prompt =  f"Considering the job description: {job_description}, and the resume provided: {resume_content}, identify areas for enhancement in the resume. Offer specific suggestions on how to improve these aspects to better match the job requirements and increase the likelihood of being selected for the position of {position_applied}."

    messages = [
    {"role": "user",
     "content": [{"type": "text",
                "text": prompt
            },
        ]
    }
]

    generated_response = model.chat(messages=messages)
    career_advice = generated_response['choices'][0]['message']['content']

    return career_advice


career_advice_app = gr.Interface(
    fn = generate_career_advice,
    flagging_mode = 'never',
    inputs = [
        gr.Textbox(label= 'Position Applied For', placeholder = "Enter the position you are applying for..."),
        gr.Textbox(label = "Job Description Information", placeholder="Paste the job description here...", lines=10),
        gr.File(label = "Resume Content", file_types = [".pdf", ".docx", ".txt"]),
    ],
    outputs = gr.Textbox(label = 'Advice', lines = 60),
    title = "Izzy's Career Advisor",
    description = "Enter the position you're applying for, paste the job description, and your resume content to get advice on what to improve for getting this job."
)

career_advice_app.launch()