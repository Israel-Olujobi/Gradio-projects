# Exercise

# A news agency publishes hundreds of articles daily on its website. 
# Each article contains several images relevant to the story. 
# Writing appropriate and descriptive captions for each image manually is a tedious task and might slow down the publication process.

#In this scenario, your image captioning program can expedite the process:

# Firstly, you send a HTTP request to the provided URL and retrieve the webpage's content. 
# This content is then parsed by BeautifulSoup

import requests
from PIL import Image
from bs4 import BeautifulSoup
from io import BytesIO
from transformers import AutoProcessor, BlipForConditionalGeneration

processor =  AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

url = "https://en.wikipedia.org/wiki/IBM"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
print('Status:', response.status_code, 'HTML size:', len(response.text))

soup = BeautifulSoup(response.text, 'html.parser')     # Scrape the url
img_elements = soup.find_all('img')                   # Derive the all the images in the url
print (f"Found {len(img_elements)} <img> tags")




with open("captions.txt", "w", encoding="utf-8") as caption_file:
    for idx, img_element in enumerate(img_elements, start=1):
        # Try different attributes
        img_url = img_element.get("src") or img_element.get("data-src")
        if not img_url and img_element.has_attr("srcset"):
            img_url = img_element["srcset"].split()[0]
        if not img_url:
            continue
        # Skip SVGs directly
        if img_url.endswith(".svg") or ".svg" in img_url:
            continue
        # Fix relative URLs
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif img_url.startswith("/"):
            img_url = "https://en.wikipedia.org" + img_url
        elif not img_url.startswith("http"):
            continue


        try:
            r = requests.get(img_url, headers=headers, timeout=10)
            raw_image = Image.open(BytesIO(r.content))               # open images content

            # Skip very small images
            if raw_image.size[0] * raw_image.size[1] < 200:
                continue                                     
 
            # continue: used to skip the rest of the loop

            raw_image = raw_image.convert('RGB')

            text = "the image of"
            inputs = processor(images=image, text=text, return_tensors='pt')
            outputs = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
    

            caption_file.write(f"{img_url}: {caption}\n")
            print(f"[{idx}] Caption saved")
    


        except OSError:
        # Skip images PIL cannot open (SVG, ICO, corrupt files)
            continue
        except Exception as e:
            print(f"[{idx}] Error: {e}")
            continue


