"""
MediPulse AI: Clinical Triage & Telemedicine Streaming Microservice
Author: Ige Fadele (https://igefadele.savadub.com)

A high-performance FastAPI microservice implementing:
1. Tier 1: Deterministic Vital Signs Emergency Acuity Gate (<2ms).
2. Tier 2: WHO Protocol-grounded Clinical Decision Routing.
3. Tier 3: Asynchronous Server-Sent Events (SSE) streaming of structured SOAP notes.
"""

from contextlib import asynccontextmanager
import os
import threading
import time
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Global model containers
model_registry = {}

BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./medipulse_clinical_adapter")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads clinical models on startup and cleans up memory on shutdown."""
    print("[MediPulse AI] Initializing Clinical Triage Microservice...")
    try:
        if os.path.exists(ADAPTER_PATH):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f"[MediPulse AI] Loading base model: {BASE_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print(f"[MediPulse AI] Attaching clinical LoRA adapter: {ADAPTER_PATH}")
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            model.eval()

            model_registry["tokenizer"] = tokenizer
            model_registry["model"] = model
            print("[MediPulse AI] Model successfully loaded into memory.")
        else:
            print(f"[MediPulse AI] Notice: Adapter at '{ADAPTER_PATH}' not detected. Running in simulated fallback mode for local evaluation.")
    except Exception as exc:
        print(f"[MediPulse AI] Engine initialization warning: {exc}")
    yield
    model_registry.clear()
    print("[MediPulse AI] Service shutdown: models unloaded.")

app = FastAPI(
    title="MediPulse AI - Clinical Triage Microservice",
    description="Deterministic vital-sign safety gating + QLoRA-streamed clinical SOAP summaries.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClinicalIntakeRequest(BaseModel):
    patient_id: str = Field(..., example="PAT-9082", description="Unique anonymized patient identifier")
    age_years: float = Field(..., ge=0.0, le=120.0, example=4.0, description="Patient age in years")
    temperature_celsius: float = Field(..., ge=32.0, le=44.0, example=39.6, description="Core body temperature in Celsius")
    heart_rate_bpm: int = Field(..., ge=25, le=250, example=135, description="Heart rate in beats per minute")
    spo2_percent: int = Field(..., ge=40, le=100, example=93, description="Blood oxygen saturation (SpO2)")
    systolic_bp: Optional[int] = Field(default=None, ge=50, le=260, example=110, description="Systolic blood pressure (mmHg)")
    is_pregnant: bool = Field(default=False, description="Whether the patient is currently pregnant")
    reported_symptoms: str = Field(
        ...,
        min_length=5,
        example="Hotness of body since yesterday, vomiting everything, breathing fast-fast",
        description="Patient or caregiver complaint (free text / regional dialect)"
    )

def evaluate_tier1_vitals(request: ClinicalIntakeRequest) -> Optional[str]:
    """
    Tier 1: Deterministic Physiological Safety Gate (<2ms).
    Immediately flags life-threatening vital sign derangements without touching the GPU.
    """
    alerts = []
    # Severe hypoxemia
    if request.spo2_percent < 90:
        alerts.append(f"Severe Hypoxemia (SpO2: {request.spo2_percent}% < 90%)")

    # Hyperpyrexia
    if request.temperature_celsius >= 40.5:
        alerts.append(f"Hyperpyrexia (Temp: {request.temperature_celsius}°C >= 40.5°C)")

    # Severe tachycardia / bradycardia (pediatric vs adult threshold)
    if request.age_years >= 12 and (request.heart_rate_bpm > 140 or request.heart_rate_bpm < 45):
        alerts.append(f"Extreme Heart Rate Derangement ({request.heart_rate_bpm} bpm)")
    elif request.age_years < 12 and (request.heart_rate_bpm > 170 or request.heart_rate_bpm < 60):
        alerts.append(f"Pediatric Heart Rate Danger Sign ({request.heart_rate_bpm} bpm)")

    # Severe hypertension in pregnancy (Impending Eclampsia danger)
    if request.is_pregnant and request.systolic_bp and request.systolic_bp >= 160:
        alerts.append(f"Hypertensive Crisis in Pregnancy (SBP: {request.systolic_bp} mmHg >= 160)")

    if alerts:
        return " | ".join(alerts)
    return None

def mock_clinical_stream(request: ClinicalIntakeRequest) -> AsyncGenerator[str, None]:
    """Simulates realistic streamed SOAP note when model weights are not loaded locally."""
    intro = (
        "data: ### CLINICAL TRIAGE INTAKE SUMMARY (SOAP)\n"
        f"data: - **Patient**: ID {request.patient_id} ({request.age_years} years old)\n"
        f"data: - **Reported Vitals**: Temp {request.temperature_celsius}°C | HR {request.heart_rate_bpm} bpm | SpO2 {request.spo2_percent}%\n"
        "data: - **Subjective**: Patient/caregiver reports acute symptoms: " + request.reported_symptoms + "\n"
        "data: - **Objective**: High fever and clinical distress noted in triage input.\n"
        "data: - **Assessment**: Acute symptomatic presentation requiring prompt physician review. Priority triage assigned based on reported vital profile.\n"
        "data: - **Triage Acuity**: PRIORITY QUEUE (YELLOW / URGENT)\n"
        "data: - **Plan for Clinician**: Confirm vitals, perform point-of-care rapid testing (mRDT / blood glucose), maintain hydration, physician bedside consult.\n\n"
    )
    for line in intro.split("\n"):
        if line:
            yield line + "\n\n"
            time.sleep(0.04)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "MediPulse AI Clinical Triage",
        "model_loaded": "model" in model_registry,
        "device": "CUDA" if os.environ.get("CUDA_VISIBLE_DEVICES") else "CPU/Simulated"
    }

@app.post("/triage/stream")
def triage_stream_endpoint(request: ClinicalIntakeRequest):
    # Step 1: Execute Tier 1 Deterministic Gate
    emergency_alert = evaluate_tier1_vitals(request)
    if emergency_alert:
        def emergency_stream():
            yield (
                f"data: [CRITICAL EMERGENCY RED-FLAG ALERT]: {emergency_alert}\n"
                f"data: ACTION: Route patient immediately to Resuscitation / Urgent Physician Bedside.\n\n"
            )
        return StreamingResponse(emergency_stream(), media_type="text/event-stream")

    # Step 2: Stream Clinical SOAP Note (Model or Simulated Stream)
    if "model" in model_registry and "tokenizer" in model_registry:
        def live_stream():
            import torch
            from transformers import TextIteratorStreamer

            tokenizer = model_registry["tokenizer"]
            model = model_registry["model"]

            system_prompt = (
                "You are MediPulse AI. Generate a structured clinical SOAP note summarizing the patient intake "
                "for the attending physician. Highlight red flags and triage acuity. NEVER prescribe medications."
            )
            user_prompt = (
                f"Patient Age: {request.age_years}yo. Vitals: Temp {request.temperature_celsius}°C, "
                f"HR {request.heart_rate_bpm} bpm, SpO2 {request.spo2_percent}%. "
                f"Symptoms: {request.reported_symptoms}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt_str], return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs = dict(inputs, streamer=streamer, max_new_tokens=400, temperature=0.2)

            thread = threading.Thread(target=model.generate, kwargs=kwargs)
            thread.start()

            for token in streamer:
                yield f"data: {token}\n\n"

        return StreamingResponse(live_stream(), media_type="text/event-stream")

    return StreamingResponse(mock_clinical_stream(request), media_type="text/event-stream")
