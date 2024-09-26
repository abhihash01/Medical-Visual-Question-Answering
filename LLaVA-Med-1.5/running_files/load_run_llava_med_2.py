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


def generate_caption(image_file,prompt, tokenizer, model, image_processor, context_len):
    if image_file.startswith('http://') or image_file.startswith('https://'):
      response = requests.get(image_file)
      image = Image.open(BytesIO(response.content)).convert('RGB')
    
    else:
      image = Image.open(image_file).convert('RGB')
    conv_mode = "mistral_instruct"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
      image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
      image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    print("the shape of image tensor is")

    
    print(image_tensor.shape)

    print("size is")
    print(image_tensor.size())
    inp = f"{roles[0]}: {prompt}"
    print(inp)
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    print(inp)
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    print("raw prompt")
    print(raw_prompt)
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    print("stop string")
    print(stop_str)
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    #streamer code
    with torch.inference_mode():
      output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                  max_new_tokens=1024, use_cache=True)
      
      print("the length of the output is")
      print(len(output_ids))
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    #output = outputs.rsplit('', 1)[0]
    print(outputs)
    output = outputs
    return image, output

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

image, output = generate_caption(f'https://raw.githubusercontent.com/microsoft/LLaVA-Med/main/llava/serve/examples/med_img_1.png', 'Tell me about the image',tokenizer, model, image_processor, context_len)
print(output)