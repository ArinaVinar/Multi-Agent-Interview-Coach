# logger.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class TurnLog:
    turn_id: int
    topic: str
    difficulty: str
    score_0_100: int
    ideal_answer_short: str

    agent_visible_message: str
    user_message: str
    internal_thoughts: str


class InterviewLogger:
    def __init__(self, participant_name: str):
        self.participant_name = participant_name
        self.turns: List[TurnLog] = []
        self.final_feedback: Optional[Dict[str, Any]] = None

    def add_turn(
        self,
        *,
        turn_id: int,
        topic: str,
        difficulty: str,
        score_0_100: int,
        ideal_answer_short: str,
        agent_msg: str,
        user_msg: str,
        internal: str,
    ) -> None:
        self.turns.append(
            TurnLog(
                turn_id=turn_id,
                topic=topic,
                difficulty=difficulty,
                score_0_100=score_0_100,
                ideal_answer_short=ideal_answer_short,
                agent_visible_message=agent_msg,
                user_message=user_msg,
                internal_thoughts=internal,
            )
        )

    def set_final_feedback(self, feedback: Dict[str, Any]) -> None:
        self.final_feedback = feedback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_name": self.participant_name,
            "turns": [asdict(t) for t in self.turns],
            "final_feedback": self.final_feedback,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
