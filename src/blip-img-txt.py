from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# https://huggingface.co/Salesforce/blip-image-captioning-base

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
# requests.get(img_url, stream=True).raw

raw_image = Image.open(
    "/dd/engine/src/client/public/cyberia/assets/ai-resources/sprites/Gemini_Generated_Image_ytdwclytdwclytdw-frame2.png"
).convert("RGB")

# conditional image captioning
text = "right, left, backwards, in this case it orientarion looks "
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
