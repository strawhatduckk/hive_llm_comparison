import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

# disable some warnings
# transformers.logging.set_verbosity_error()
# transformers.logging.disable_progress_bar()
# warnings.filterwarnings('ignore')

IMAGE_POLICY = """
The following is our company's content moderation policy, based on the moderation policy, we gather image information from the user uploaded image. Please answer the question with json format.

1. Explicit Nudity: it contains Nudity, Graphic Male Nudity, Graphic Female Nudity, Sexual Activity, Illustrated Explicit Nudity and Adult Toys.
2. Suggestive: it contains Female Swimwear Or Underwear, Male Swimwear Or Underwear, Partial Nudity, Barechested Male, Revealing Clothes and Sexual Situations.
3. Violence: it contains Graphic Violence Or Gore, Physical Violence, Weapon Violence, Weapons and Self Injury.
4. Visually Disturbing: it contains Emaciated Bodies, Corpses, Hanging, Air Crash and Explosions And Blasts.
5. Rude Gestures: it contains Middle Finger.
6. Hate Symbols: it contains Nazi Party, White Supremacy and Extremist.
"""

IMAGE_QUESTION = "Based on the Moderation policy above, tell me if the image contains unsafe content, also give its category and reason if it's unsafe. Please anwser the question with the following format and only put explanation into the reason field: "
IMAGE_QUESTION += """
{
    "flag": "xxx",
    "category": "xxx",
    "reason": "the reason is ..."
}
"""

class Bunny:
    def __init__(self) -> None:
        if torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"
        torch.set_default_device(self.device)
        # create model
        self.model = AutoModelForCausalLM.from_pretrained(
            "BAAI/Bunny-v1_1-Llama-3-8B-V",
            torch_dtype=torch.float32, # float16 might be faster for GPUs since it's smaller and also uses less memory but many CPUs don't natively support FP16 arithmetic, so it's all being done in software => slow
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/Bunny-v1_1-Llama-3-8B-V",
            trust_remote_code=True,
            use_fast=True
        )


    def moderate_image(self, image):
        input_text = f"{IMAGE_POLICY}\nUSER: <image>\n{IMAGE_QUESTION} ASSISTANT:"
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in input_text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(self.device)
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype, device=self.device)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=100,
            use_cache=True,
            repetition_penalty=1.0 # increase this to avoid chattering
        )[0]

        return self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

image = Image.open('img.jpg')
bunny = Bunny()
res = bunny.moderate_image(image)
print(res)