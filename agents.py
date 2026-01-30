# agents.py
from __future__ import annotations
from typing import Dict, List, Literal, Any, Optional
from pydantic import BaseModel, Field, ValidationError
import json

from ollama_client import OllamaClient
from prompts import (
    OBSERVER_SYSTEM,
    INTERVIEWER_SYSTEM,
    HIRING_MANAGER_SYSTEM,
    QUESTION_GENERATOR_SYSTEM,
)

Difficulty = Literal["easy", "medium", "hard"]


def _safe_json_load(s: str) -> Any:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
        raise


class TopicPlan(BaseModel):
    topics: List[str] = Field(default_factory=list)


class ObserverOutput(BaseModel):
    detected_offtopic: bool = False
    detected_hallucination: bool = False

    answer_quality: Literal["good", "partial", "poor", "unknown"] = "unknown"
    score_0_100: int = Field(ge=0, le=100, default=50)

    next_difficulty: Difficulty = "easy"
    next_topic: str = "general_basics"
    intent: str = "check_basics"

    should_move_on: bool = True
    need_hint: bool = False
    acknowledge: str = "Понял."
    notes: str = ""


class GeneratedQuestion(BaseModel):
    question_text: str
    hint: str = ""
    ideal_answer_short: str = ""


class FinalFeedback(BaseModel):
    grade: Literal["Junior", "Middle", "Senior"]
    hiring_recommendation: Literal["Hire", "No Hire", "Strong Hire"]
    confidence_score_0_100: int = Field(ge=0, le=100)

    confirmed_skills: List[str]
    knowledge_gaps: List[str]
    corrections: List[str]

    clarity: Literal["low", "medium", "high"]
    honesty: Literal["low", "medium", "high"]
    engagement: Literal["low", "medium", "high"]

    roadmap: List[str]


class TopicPlannerAgent:
    """Generates a topic plan once for the given position/grade/experience."""

    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def build_plan(self, *, position: str, grade: str, experience: str, lang: str = "ru") -> TopicPlan:
        schema_hint = {"topics": ["string", "..."]}

        user_prompt = f"""
Create an interview topic plan for:
- Position: {position}
- Grade: {grade}
- Experience: {experience}
Language: {lang}

Rules:
- Topics must be skill-focused (knowledge checks), not only "tell about your project".
- Output 6-10 topics from fundamentals to more advanced, appropriate for the grade.
- Topics should be short identifiers like:
  python_collections, java_concurrency, linux_permissions, networking_dns, sql_joins, git_workflow, etc.
  - MUST stay within the given position tech stack.
- If Position includes "Java" or experience mentions Spring: DO NOT ask about Python-specific concepts (list/tuple, Python syntax, etc.).
- Use Java/Spring terminology for examples when applicable.

Return ONLY JSON:
{schema_hint}
""".strip()

        messages = [
            {"role": "system", "content": "You create topic plans for technical interviews. Return JSON only."},
            {"role": "user", "content": user_prompt},
        ]

        raw = self.client.chat(self.model, messages, format="json", options={"temperature": 0.3})
        try:
            data = _safe_json_load(raw)
            plan = TopicPlan.model_validate(data)
            # fallback safety
            if not plan.topics:
                plan.topics = ["general_basics", "problem_solving", "tools_workflow"]
            return plan
        except Exception:
            return TopicPlan(topics=["general_basics", "problem_solving", "tools_workflow"])


class ObserverAgent:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def analyze(
        self,
        *,
        position: str,
        grade: str,
        experience: str,
        history: List[Dict[str, str]],
        last_user: str,
        state: Dict[str, Any],
        current_topic: str,
        current_difficulty: Difficulty,
        topic_plan: List[str],
    ) -> ObserverOutput:
        schema_hint = {
            "detected_offtopic": False,
            "detected_hallucination": False,
            "answer_quality": "good|partial|poor|unknown",
            "score_0_100": 0,
            "next_difficulty": "easy|medium|hard",
            "next_topic": "string",
            "intent": "string",
            "should_move_on": True,
            "need_hint": False,
            "acknowledge": "string",
            "notes": "string",
        }

        user_prompt = f"""
You are the Observer/Critic. Analyze the candidate reply and decide the next step.

Context:
- Position: {position}
- Target grade: {grade}
- Candidate experience: {experience}
- Current topic: {current_topic}
- Current difficulty: {current_difficulty}
- Topic plan (use these topics, avoid random unrelated ones): {topic_plan}

State (avoid repeats):
{state}

Candidate last message:
{last_user}

Tasks:
1) Detect off-topic and hallucinations (nonsense confident claims).
2) Evaluate answer quality, give score 0-100.
3) Decide next difficulty:
   - good: increase or keep, maybe move on
   - poor/unknown: simplify and set need_hint=true
4) Decide next topic:
   - If should_move_on=false => keep current_topic
   - Else choose next topic from topic_plan (prefer the next one in order)
5) Provide short acknowledge (1 sentence, natural).

Return ONLY JSON:
{schema_hint}
""".strip()

        messages = [{"role": "system", "content": OBSERVER_SYSTEM}]
        messages += history[-14:]
        messages.append({"role": "user", "content": user_prompt})

        raw = self.client.chat(self.model, messages, format="json", options={"temperature": 0.2})
        try:
            data = _safe_json_load(raw)
            out = ObserverOutput.model_validate(data)
            # guard: if next_topic not in plan and should_move_on=True -> stay in plan
            if out.should_move_on and out.next_topic not in topic_plan:
                out.next_topic = current_topic
                out.should_move_on = False
            return out
        except Exception:
            return ObserverOutput(
                answer_quality="unknown",
                score_0_100=45,
                next_difficulty=current_difficulty,
                next_topic=current_topic,
                intent="fallback",
                should_move_on=False,
                need_hint=True,
                acknowledge="Окей, понял.",
                notes=f"Observer parse failed. Raw head: {raw[:180]}",
            )


class QuestionGeneratorAgent:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def generate(
        self,
        *,
        position: str,
        grade: str,
        experience: str,
        topic: str,
        difficulty: Difficulty,
        intent: str,
        need_hint: bool,
        avoid_questions: List[str],
        lang: str = "ru",
    ) -> GeneratedQuestion:
        schema_hint = {
            "question_text": "string",
            "hint": "string (optional)",
            "ideal_answer_short": "string (1-4 sentences)",
        }

        user_prompt = f"""
Generate ONE interview question.

Inputs:
- Language: {lang}
- Position: {position}
- Grade: {grade}
- Experience: {experience}
- Topic: {topic}
- Difficulty: {difficulty}
- Intent: {intent}
- Need hint: {need_hint}
- Avoid repeating these questions (paraphrase or choose different angle): {avoid_questions}

Rules:
- The question must match the topic and difficulty.
- If need_hint=true: provide a short hint (1 sentence) that helps but doesn't give full answer.
- Provide ideal_answer_short: a brief correct answer (1-4 sentences) suitable for feedback.
- Return ONLY JSON:
{schema_hint}
""".strip()

        messages = [{"role": "system", "content": QUESTION_GENERATOR_SYSTEM}]
        messages.append({"role": "user", "content": user_prompt})

        raw = self.client.chat(self.model, messages, format="json", options={"temperature": 0.4})
        try:
            data = _safe_json_load(raw)
            q = GeneratedQuestion.model_validate(data)
            if not q.question_text:
                q.question_text = "Расскажи, чем отличаются list и tuple в Python?"
            if not need_hint:
                q.hint = ""
            return q
        except Exception:
            return GeneratedQuestion(
                question_text="Давай начнём с базы: чем list отличается от tuple?",
                hint="Подумай про изменяемость (mutability). " if need_hint else "",
                ideal_answer_short="list изменяемый, tuple неизменяемый; tuple можно использовать там, где важна неизменность.",
            )


class InterviewerAgent:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def render_message(
        self,
        *,
        history: List[Dict[str, str]],
        observer_plan: ObserverOutput,
        last_user: str,
        question_text: str,
        hint: str,
    ) -> str:
        user_prompt = f"""
You are the Interviewer. Produce the next visible message in Russian (unless the conversation is clearly in English).

Observer plan:
- acknowledge: {observer_plan.acknowledge}
- detected_offtopic: {observer_plan.detected_offtopic}
- detected_hallucination: {observer_plan.detected_hallucination}
- need_hint: {observer_plan.need_hint}
- notes: {observer_plan.notes}

Candidate last message:
{last_user}

Question to ask next:
{question_text}

Hint (use only if need_hint=true):
{hint}

Rules:
- Start with acknowledge (1 sentence).
- If off-topic: gently steer back in 1 sentence, then ask the question.
- If hallucination: politely correct in 1-2 sentences, then ask the question.
- If candidate asked a relevant question about role/company/stack: answer briefly (2-4 sentences), then ask the question.
- If need_hint=true: add "Подсказка:" + hint after the question.
- Keep it natural, not robotic. Avoid repeating identical phrasing.
""".strip()

        messages = [{"role": "system", "content": INTERVIEWER_SYSTEM}]
        messages += history[-14:]
        messages.append({"role": "user", "content": user_prompt})

        text = self.client.chat(self.model, messages, options={"temperature": 0.5})
        return text.strip()


class HiringManagerAgent:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def build_feedback(
        self,
        *,
        position: str,
        target_grade: str,
        experience: str,
        turns: List[Dict[str, Any]],
        transcript: List[Dict[str, str]],
    ) -> FinalFeedback:
        schema_hint = {
            "grade": "Junior|Middle|Senior",
            "hiring_recommendation": "Hire|No Hire|Strong Hire",
            "confidence_score_0_100": 0,
            "confirmed_skills": ["..."],
            "knowledge_gaps": ["..."],
            "corrections": ["..."],
            "clarity": "low|medium|high",
            "honesty": "low|medium|high",
            "engagement": "low|medium|high",
            "roadmap": ["..."],
        }

        user_prompt = f"""
Position: {position}
Target grade: {target_grade}
Experience: {experience}

Turns (topic, difficulty, score, ideal_answer_short):
{turns}

Transcript:
{transcript}

Create final feedback:
- Decision: grade/hiring/confidence
- Hard skills: confirmed skills + gaps
- For each gap: include correct short explanation (use ideal_answer_short as ground truth)
- Soft skills: clarity/honesty/engagement
- Roadmap: concrete next steps

Return ONLY JSON:
{schema_hint}
""".strip()

        messages = [{"role": "system", "content": HIRING_MANAGER_SYSTEM}]
        messages.append({"role": "user", "content": user_prompt})

        raw = self.client.chat(self.model, messages, format="json", options={"temperature": 0.2})
        try:
            data = _safe_json_load(raw)
            return FinalFeedback.model_validate(data)
        except ValidationError:
            return FinalFeedback(
                grade="Junior",
                hiring_recommendation="No Hire",
                confidence_score_0_100=45,
                confirmed_skills=[],
                knowledge_gaps=["Не удалось корректно сформировать финальный отчёт (ошибка валидации)."],
                corrections=[raw[:300]],
                clarity="medium",
                honesty="medium",
                engagement="low",
                roadmap=["Повторить базовые темы и попробовать заново."],
            )
