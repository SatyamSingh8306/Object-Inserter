from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from PIL import Image,ImageOps
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os

load_dotenv()

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id = "timbrooks/instruct-pix2pix",
    huggingfacehub_api_token= HF_API_TOKEN,
    temperature=0.5,
)


# prompt = PromptTemplate(
#     template = 
#     input_variables=[]
# )]
path = "er.png"
image = Image.open(path)
image = ImageOps.exif_transpose(image)
image = image.convert("RGB")


llm.invoke(**kwargs={
    input : "remove the image logo",
    image: image,
    num_inference_steps:10,
    image_guidance_scale:1

})
ans = llm.invoke(
    input = "remove the image logo",
    image= image,
    num_inference_steps=10,
    image_guidance_scale=1)

plt.imshow(ans[0])

                          