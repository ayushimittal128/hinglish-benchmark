# src/models.py
import os, time, re
from typing import List, Dict, Optional

from google import genai
from google.genai import types

OK_LABEL = re.compile(r"[A-Za-z_ ]+")

def _normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("-", " ").replace("/", " ").replace(".", "")
    return " ".join(s.split())

class GeminiClient:
    """
    Minimal wrapper around google-genai.
    """
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-001",
        temperature: float = 0.0,
        max_retries: int = 3
    ):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment/.env")
        # Build the client (Gemini Developer API by default)
        self.client = genai.Client(api_key=api_key)  # picks up key directly
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

    def classify(self, text: str, task: str, labels: List[str]) -> Dict[str, Optional[str]]:
        """
        Returns: {"label": <pred>|None, "raw": <raw_text>|None, "latency_ms": int, "error": None|str}
        """
        labels_norm = [_normalize_label(l) for l in labels]

        system_prompt = (
            f"You are a strict classifier for Hinglish text.\n\n"
            f"TASK: {task}\n"
            f"Allowed labels (return EXACTLY one of these): {', '.join(labels_norm)}\n\n"
            "Rules:\n"
            "- Output ONLY one label from the allowed set. No explanations.\n"
            "- If unsure, pick the closest label.\n"
            "- Use lower-case words separated by single spaces.\n"
        )

        user_content = f"TEXT:\n{text}\n\nReturn exactly one label from: {labels_norm}"

        delay = 1.0
        start = time.time()

        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[user_content],  # SDK will coerce strings to user content
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                    ),
                )
                raw = (resp.text or "").strip()
                raw_norm = _normalize_label(re.sub(r"[^a-z ]", "", raw))

                if raw_norm in labels_norm:
                    pred = raw_norm
                else:
                    # try substring match; else default to first label to stay deterministic
                    candidates = [l for l in labels_norm if l in raw_norm.split()]
                    pred = candidates[0] if candidates else (labels_norm[0] if labels_norm else None)

                latency_ms = int((time.time() - start) * 1000)
                return {"label": pred, "raw": raw, "latency_ms": latency_ms, "error": None}

            except Exception as e:
                err = str(e)
                if attempt == self.max_retries - 1:
                    latency_ms = int((time.time() - start) * 1000)
                    return {"label": None, "raw": None, "latency_ms": latency_ms, "error": err}
                time.sleep(delay)
                delay *= 2
