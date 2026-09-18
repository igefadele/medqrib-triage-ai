"""
MedQrib Triage AI: Clinical LoRA Fine-Tuning Pipeline
Author: Ige Fadele (https://igefadele.savadub.com)

This script fine-tunes Meta-Llama-3-8B-Instruct on clinical triage intake
dialogues using 4-bit NormalFloat QLoRA via Hugging Face PEFT & TRL.
"""

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer

BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "medipulse_sample_intake.jsonl")
OUTPUT_ADAPTER_DIR = os.getenv("OUTPUT_ADAPTER_DIR", "./medipulse_clinical_adapter")

def train():
    print(f"[MedQrib Triage AI] Initializing QLoRA Fine-Tuning for: {BASE_MODEL_NAME}")
    print(f"[MedQrib Triage AI] Dataset source: {DATASET_PATH}")

    # 1. 4-bit NormalFloat Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True
    )

    # 2. Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Base Model Loading (with 4-bit quantization on GPU or CPU fallback)
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Adapter Hyperparameters
    lora_config = LoraConfig(
        r=16,                                    # Rank dimension
        lora_alpha=32,                           # Alpha scaling parameter
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Dataset Ingestion
    dataset = load_dataset("json", data_files=DATASET_PATH)["train"]

    def format_chat(sample):
        return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}

    formatted_dataset = dataset.map(format_chat)

    # 6. Training Configuration
    training_args = TrainingArguments(
        output_dir=OUTPUT_ADAPTER_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"
    )

    # 7. Supervised Fine-Tuning Execution
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args
    )

    print("[MedQrib Triage AI] Starting training loop...")
    trainer.train()

    print(f"[MedQrib Triage AI] Saving adapter weights to: {OUTPUT_ADAPTER_DIR}")
    model.save_pretrained(OUTPUT_ADAPTER_DIR)
    tokenizer.save_pretrained(OUTPUT_ADAPTER_DIR)
    print("[MedQrib Triage AI] Training completed successfully.")

if __name__ == "__main__":
    train()
