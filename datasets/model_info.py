from huggingface_hub import model_info

info = model_info("Qwen/Qwen2-VL-7B-Instruct")
print(info)