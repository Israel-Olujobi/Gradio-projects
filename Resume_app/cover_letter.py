from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import gradio as gr


api_key = "IYXxmg5tr-G4r-XWKOvsZqjRIToEgnS4CCkSpV5QQmsv"
project_id = "c20516cb-d9bd-4049-ae9e-8e5dc9f2e060"
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

def generate_cover_letter(company_name, position_name, resume_content, job_description):
        
    prompt =  f"Generate a customized cover letter using the company name: {company_name}, the position applied for: {position_name}, and the job description: {job_description}. Ensure the cover letter highlights my qualifications and experience as detailed in the resume content: {resume_content}. Adapt the content carefully to avoid including experiences not present in my resume but mentioned in the job description. The goal is to emphasize the alignment between my existing skills and the requirements of the role."

    messages = [
    {"role": "user",
     "content": [{"type": "text",
                "text": prompt
            },
        ]
    }
]

    generated_response = model.chat(messages=messages)
    cover_letter = generated_response['choices'][0]['message']['content']

    return cover_letter


cover_letter_app = gr.Interface(
    fn = generate_cover_letter,
    flagging_mode = 'never',
    inputs = [
        gr.Textbox(label= 'Company Name', placeholder = "Enter the name of the company..."),
        gr.Textbox(label= 'Position Name', placeholder = "Enter the name of the position..."),
        gr.Textbox(label = "Job Description Information", placeholder="Paste the job description here...", lines=10),
        gr.File(label = "Resume Content", file_types = [".pdf", ".docx", ".txt"]),
    ],
    outputs = gr.Textbox(label = 'Customized Cover Letter', lines = 60),
    title = "Izzy's Cover Letter Generator",
    description = "Generate a customized cover letter by entering the company name, position name, job description and your resume."
)

cover_letter_app.launch()