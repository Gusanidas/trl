import torch
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Define model parameters
max_seq_length = 1024
max_prompt_length = 256

# Set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 instead of bfloat16 for MPS
    device_map="auto",
)

# Define formatting tokens for reasoning and solutions
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# Create system prompt
system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# Load and process GSM8K dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")

# Extract hash answers from the dataset
def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

# Map the dataset to the required format
dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x["question"]},
    ],
    "answer": extract_hash_answer(x["answer"]),
})

# Define regex patterns for reward functions
match_format = re.compile(
    rf"^[\s]{{0,}}{reasoning_start}.+?{reasoning_end}.*?{solution_start}(.+?){solution_end}[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL
)

# Define reward functions
def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        if guess == true_answer:
            score += 3.0
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1: score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.25
                else: score -= 1.0
            except:
                score -= 0.5
        scores.append(score)
    return scores

def check_numbers(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip())
            scores.append(1.5 if guess == true_answer else 0.0)
        except:
            scores.append(0)
    return scores

# Configure GRPO training arguments
training_args = GRPOConfig(
    learning_rate=1e-6,  # Lower learning rate for full fine-tuning
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # Increased for more stability
    num_generations=2,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=50,  # Increase for a full training run
    save_steps=25,
    max_grad_norm=1.0,
    report_to="none",
    output_dir="outputs",
    fp16=False,  # Disable mixed precision training for MPS
)

# Initialize and run the trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# Clean up memory
import gc
gc.collect()
torch.cuda.empty_cache()

# Save the model
model.save_pretrained("gemma-3-finetuned")
tokenizer.save_pretrained("gemma-3-finetuned")

# Optional: Test inference
def run_inference(question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        temperature=1.0, 
        top_p=0.95, 
        top_k=64,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Uncomment to run test inference
# test_result = run_inference("What is the sqrt of 101?")
# print(test_result)

# Uncomment for GGUF conversion using HuggingFace's optimum-cli (separate installation)
# !optimum-cli export gguf gemma-3-finetuned/ --outfile gemma-3-finetuned.gguf --quantization q8_0