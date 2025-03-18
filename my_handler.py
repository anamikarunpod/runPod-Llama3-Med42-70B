import runpod
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define model path inside the container
MODEL_PATH = "/app/model"

# Load the pre-downloaded model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

print("Model is ready for use")

def handler(event, context=None):
    # Parse input
    try:
        body = event.get("input", {})
        prompt = body.get("prompt", "Explain the benefits of AI in healthcare.")
        max_length = body.get("max_length", 512)
        temperature = body.get("temperature", 0.7)
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": "Invalid JSON input."}

    if not prompt:
        return {"statusCode": 400, "body": "Prompt is required."}

    # Generate text response
    try:
        print(f"Generating response for prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs, 
            max_length=max_length, 
            do_sample=True, 
            temperature=temperature
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"response": response_text})
        }

    except Exception as e:
        print(f"Error generating response: {e}")
        return {"statusCode": 500, "body": str(e)}

# Debugging locally
if __name__ == "__main__":
    runpod.serverless.start({'handler': handler})
