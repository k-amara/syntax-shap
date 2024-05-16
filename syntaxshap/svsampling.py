import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys
from huggingface_hub import login
login(token="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")
import models

from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    ShapleyValueSampling,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
    ProductBaselines,
)

"""def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config"""


"""model_name = "meta-llama/Llama-2-13b-chat-hf" 

bnb_config = create_bnb_config()

model, tokenizer = load_model(model_name, bnb_config)
print("model loaded")

eval_prompt = "Dave lives in Palm Coast, FL and is a lawyer. His personal interests include"

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
print("model_input", model_input)
model.eval()
with torch.no_grad():
    output_ids = model.generate(model_input["input_ids"], max_new_tokens=4)[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response)"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_load = "gpt2"
tokenizer_load = "gpt2"
# Load GPT-2 model
model = AutoModelForCausalLM.from_pretrained(model_load)
model.to(device)
# Set text generation parameters
model.config.task_specific_params["text-generation"] = {
    "do_sample": True,
    "max_new_tokens": 1,
    "temperature": 0.7,
    "top_k": 50,
    #"no_repeat_ngram_size": 2,
}
model.config.is_decoder = True

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_load)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Initialize TextGeneration model
lmmodel = models.TextGeneration(model, tokenizer, device=device)

sv = ShapleyValueSampling(model) 

llm_attr = LLMAttribution(sv,lmmodel.tokenizer)


eval_prompt = "Dave lives in Palm Coast, FL and is a lawyer. His personal interests include"

inp = TextTokenInput(
    eval_prompt, 
    lmmodel.tokenizer,
    skip_tokens=[1],  # skip the special token for the start of the text <s>
)
print("inpt", inp)

target = "playing guitar."

attr_res = llm_attr.attribute(inp, target=target)

print("attr to the output sequence:", attr_res.seq_attr.shape)  # shape(n_input_token)
print("attr to the output tokens:", attr_res.token_attr.shape)  # shape(n_output_token, n_input_token)