from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# مسیر به پوشه مدل که فایل‌ها داخلش هست
model_dir = "/path/to/your/model/directory"

# بارگذاری توکنایزر و مدل
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# متن ورودی برای ادامه دادن
prompt = "Once upon a time, in a land far away,"

# تبدیل ورودی به توکن‌ها
inputs = tokenizer(prompt, return_tensors="pt")

# تولید ادامه متن
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# تبدیل خروجی به متن
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
