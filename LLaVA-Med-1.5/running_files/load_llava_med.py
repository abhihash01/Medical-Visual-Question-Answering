import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("the device is ")
print(device)

model_path= model_path='microsoft/llava-med-v1.5-mistral-7b'
model_base=None
model_name='llava-med-v1.5-mistral-7b'
load_4bit=True
##device='cuda'
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device="cuda")

print("done")
model.to(device)

#tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, args.load_4bit, device=args.device)


print("loaded the model")