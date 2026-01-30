# interview_engine.py
from __future__ import annotations

from typing import Dict, List, Any

from agents import (
    ObserverAgent,
    InterviewerAgent,
    HiringManagerAgent,
    TopicPlannerAgent,
    QuestionGeneratorAgent,
    ObserverOutput,
    Difficulty,
)
from logger import InterviewLogger
from prompts import build_interviewer_intro


class InterviewEngine:
    """
    Multi-agent interview orchestrator.

    Key goals:
    - Single source of truth for "planned_topic" (prevents 'said ML but asked SQL' mismatch).
    - Dynamic difficulty via streaks.
    - Anti-repeat via passing recent asked questions to the generator.
    - Optional universal topic-consistency guard: re-ask generator if it drifted from topic (no hardcoded markers).
    """

    def __init__(
        self,
        *,
        position: str,
        target_grade: str,
        experience: str,
        participant_name: str,
        interviewer: InterviewerAgent,
        observer: ObserverAgent,
        hiring_manager: HiringManagerAgent,
        planner: TopicPlannerAgent,
        qgen: QuestionGeneratorAgent,
        lang: str = "ru",
        max_regen_attempts: int = 1,
    ):
        self.position = position
        self.target_grade = target_grade
        self.experience = experience
        self.participant_name = participant_name
        self.lang = lang
        self.max_regen_attempts = max_regen_attempts

        self.interviewer = interviewer
        self.observer = observer
        self.hiring_manager = hiring_manager
        self.planner = planner
        self.qgen = qgen

        self.logger = InterviewLogger(participant_name)
        self.turn_id = 0
        self.history: List[Dict[str, str]] = []

        # --- state machine slots ---
        self.state: Dict[str, Any] = {
            "last_topic": None,
            "last_intent": None,
            "topic_index": 0,
            "streak_good": 0,
            "streak_poor": 0,
        }

        # --- dynamic topic plan ---
        plan = self.planner.build_plan(
            position=position, grade=target_grade, experience=experience, lang=lang
        )
        self.topic_plan: List[str] = plan.topics or ["general_basics", "problem_solving", "tools_workflow"]

        self.current_topic: str = self.topic_plan[0]
        self.difficulty: Difficulty = "easy"

        # anti-repeat storage
        self.asked_questions_text: List[str] = []
        self.asked_topics: List[str] = []

        # intro
        intro = build_interviewer_intro(position, target_grade, experience)
        self._push_assistant(intro)

        # ask first generated question
        first_q = self._generate_question(
            topic=self.current_topic,
            difficulty=self.difficulty,
            intent="warmup",
            need_hint=False,
        )
        self.asked_questions_text.append(first_q["question_text"])
        self.asked_topics.append(self.current_topic)
        self._push_assistant(first_q["question_text"])

    def _push_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def _push_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def get_all_assistant_messages(self) -> List[str]:
        return [m["content"] for m in self.history if m["role"] == "assistant"]

    def _update_difficulty_by_streaks(self) -> None:
        # Increase difficulty after two strong answers
        if self.state["streak_good"] >= 2:
            if self.difficulty == "easy":
                self.difficulty = "medium"
            elif self.difficulty == "medium":
                self.difficulty = "hard"
            self.state["streak_good"] = 0

        # Decrease difficulty after two weak answers
        if self.state["streak_poor"] >= 2:
            if self.difficulty == "hard":
                self.difficulty = "medium"
            elif self.difficulty == "medium":
                self.difficulty = "easy"
            self.state["streak_poor"] = 0

    def _advance_topic(self) -> str:
        i = self.state["topic_index"]
        if i < len(self.topic_plan) - 1:
            self.state["topic_index"] += 1
        return self.topic_plan[self.state["topic_index"]]

    def _topic_guard_ok(self, topic: str, question_text: str) -> bool:
        """
        Universal, non-hardcoded topic consistency check.

        We don't use keywords (since topics can be anything).
        Instead, we ask the Observer LLM quickly (cheap) to validate that the question matches the topic.
        This keeps the system robust for arbitrary topics.

        If you want zero extra LLM calls, return True here.
        """
        try:
            # If your ObserverAgent doesn't support a separate check method,
            # we use analyze() with a special instruction via "state" and "last_user".
            # This is a pragmatic approach without adding new files/classes.
            check_prompt = (
                f"TOPIC_GUARD: Does the following question strictly match the topic?\n"
                f"TOPIC: {topic}\n"
                f"QUESTION: {question_text}\n"
                f"Answer ONLY with JSON: {{\"ok\": true/false, \"reason\": \"...\"}}"
            )

            out: ObserverOutput = self.observer.analyze(
                position=self.position,
                grade=self.target_grade,
                experience=self.experience,
                history=self.history,
                last_user=check_prompt,
                state={
                    **self.state,
                    "mode": "topic_guard",
                    "topic": topic,
                    "question": question_text,
                },
                current_topic=topic,
                current_difficulty=self.difficulty,
                topic_plan=self.topic_plan,
            )
            # We can't rely on ObserverOutput schema for ok/reason,
            # so default to "ok" unless clear mismatch is detected in notes.
            notes = (out.notes or "").lower()
            # Heuristic: if observer explicitly says mismatch in notes -> fail
            if "mismatch" in notes or "не соответствует" in notes or "не по теме" in notes:
                return False
            return True
        except Exception:
            # If guard fails, don't block the interview.
            return True

    def _generate_question(self, *, topic: str, difficulty: Difficulty, intent: str, need_hint: bool) -> Dict[str, str]:
        """
        Generates a question via QGen. Optionally re-generates if topic drift is detected (universal guard).
        Returns dict with: question_text, hint, ideal_answer_short.
        """
        avoid = self.asked_questions_text[-8:]  # avoid last few

        gen = self.qgen.generate(
            position=self.position,
            grade=self.target_grade,
            experience=self.experience,
            topic=topic,
            difficulty=difficulty,
            intent=intent,
            need_hint=need_hint,
            avoid_questions=avoid,
            lang=self.lang,
        )

        # Optional universal guard (no hardcoded markers)
        # If you want to avoid extra LLM calls, set max_regen_attempts=0.
        attempts = 0
        while attempts < self.max_regen_attempts and not self._topic_guard_ok(topic, gen.question_text):
            attempts += 1
            gen = self.qgen.generate(
                position=self.position,
                grade=self.target_grade,
                experience=self.experience,
                topic=topic,
                difficulty=difficulty,
                intent=f"{intent}_REGEN_TOPIC_MISMATCH",
                need_hint=need_hint,
                avoid_questions=avoid + [gen.question_text],
                lang=self.lang,
            )

        return {
            "question_text": gen.question_text,
            "hint": gen.hint,
            "ideal_answer_short": gen.ideal_answer_short,
        }

    def step(self, user_message: str) -> str:
        self.turn_id += 1
        self._push_user(user_message)

        # 1) Observer анализирует (hidden reflection)
        plan: ObserverOutput = self.observer.analyze(
            position=self.position,
            grade=self.target_grade,
            experience=self.experience,
            history=self.history,
            last_user=user_message,
            state={
                **self.state,
                "asked_topics": self.asked_topics[-8:],
                "asked_questions_sample": self.asked_questions_text[-5:],
            },
            current_topic=self.current_topic,
            current_difficulty=self.difficulty,
            topic_plan=self.topic_plan,
        )

        # 2) Update streaks for adaptability
        if plan.score_0_100 >= 75:
            self.state["streak_good"] += 1
            self.state["streak_poor"] = 0
        elif plan.score_0_100 <= 45 or plan.answer_quality in {"poor", "unknown"}:
            self.state["streak_poor"] += 1
            self.state["streak_good"] = 0

        self._update_difficulty_by_streaks()

        # 3) Decide planned topic for THIS turn (single source of truth)
        if not plan.should_move_on:
            planned_topic = self.current_topic
            # If struggling, simplify and allow hint
            if plan.score_0_100 <= 45:
                self.difficulty = "easy"
                plan.need_hint = True
        else:
            if plan.next_topic in self.topic_plan:
                planned_topic = plan.next_topic
                self.state["topic_index"] = self.topic_plan.index(plan.next_topic)
            else:
                planned_topic = self._advance_topic()
                self.state["topic_index"] = self.topic_plan.index(planned_topic)

        # Commit topic for this turn
        self.current_topic = planned_topic

        # 4) Generate next question strictly for planned_topic
        gen = self._generate_question(
            topic=planned_topic,
            difficulty=self.difficulty,
            intent=plan.intent,
            need_hint=plan.need_hint,
        )

        self.asked_questions_text.append(gen["question_text"])
        self.asked_topics.append(planned_topic)

        # 5) Internal log (readable)
        internal = (
            f"[Observer] planned_topic={planned_topic}, quality={plan.answer_quality}, score={plan.score_0_100}, "
            f"offtopic={plan.detected_offtopic}, hallucination={plan.detected_hallucination}, "
            f"should_move_on={plan.should_move_on}, need_hint={plan.need_hint}, difficulty_now={self.difficulty}. "
            f"notes={plan.notes}"
        )

        # 6) Visible message
        visible = self.interviewer.render_message(
            history=self.history,
            observer_plan=plan,
            last_user=user_message,
            question_text=gen["question_text"],
            hint=gen["hint"],
        )
        self._push_assistant(visible)

        # 7) Structured logging
        self.logger.add_turn(
            turn_id=self.turn_id,
            topic=planned_topic,
            difficulty=self.difficulty,
            score_0_100=plan.score_0_100,
            ideal_answer_short=gen["ideal_answer_short"],
            agent_msg=visible,
            user_msg=user_message,
            internal=internal,
        )

        # 8) Update state
        self.state["last_topic"] = planned_topic
        self.state["last_intent"] = plan.intent

        return visible

    def finish(self) -> dict:
        turns_dict = [t.__dict__ for t in self.logger.turns]
        feedback = self.hiring_manager.build_feedback(
            position=self.position,
            target_grade=self.target_grade,
            experience=self.experience,
            turns=turns_dict,
            transcript=self.history,
        )
        feedback_dict = feedback.model_dump()
        self.logger.set_final_feedback(feedback_dict)
        return feedback_dict
