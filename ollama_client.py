# ollama_client.py
from __future__ import annotations
import requests
from typing import Any, Dict, List, Optional


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Calls Ollama /api/chat and returns assistant content as string.
        If format="json", Ollama tries to output valid JSON.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if format:
            payload["format"] = format
        if options:
            payload["options"] = options

        url = f"{self.base_url}/api/chat"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]
