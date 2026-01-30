# main.py
from __future__ import annotations
import argparse
import json
from datetime import datetime

from ollama_client import OllamaClient
from agents import (
    ObserverAgent,
    InterviewerAgent,
    HiringManagerAgent,
    TopicPlannerAgent,
    QuestionGeneratorAgent,
)
from interview_engine import InterviewEngine

from scenarios import SCENARIOS


def run_interactive(args):
    client = OllamaClient(base_url=args.ollama_url)

    interviewer = InterviewerAgent(client, model=args.interviewer_model)
    observer = ObserverAgent(client, model=args.observer_model)
    hiring_manager = HiringManagerAgent(client, model=args.manager_model)

    planner = TopicPlannerAgent(client, model=args.observer_model)  # можно тем же, что observer
    qgen = QuestionGeneratorAgent(client, model=args.interviewer_model)  # или отдельной моделью

    engine = InterviewEngine(
        position=args.position,
        target_grade=args.grade,
        experience=args.experience,
        participant_name=args.name,
        interviewer=interviewer,
        observer=observer,
        hiring_manager=hiring_manager,
        planner=planner,
        qgen=qgen,
        lang="ru",
    )

    print("\n=== Interview started ===")
    print("\n".join(engine.get_all_assistant_messages()))
    print('Напишите "стоп" чтобы завершить.\n')

    while True:
        user_msg = input("YOU> ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"стоп", "stop", "стоп интервью"}:
            break

        bot_msg = engine.step(user_msg)
        print(f"\nBOT> {bot_msg}\n")

    feedback = engine.finish()
    print("\n=== FINAL FEEDBACK (JSON) ===")

    print(json.dumps(feedback, ensure_ascii=False, indent=2))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"logs/interview_{ts}.json"
    engine.logger.save(out_path)
    print(f"\nSaved log to: {out_path}")


def run_scenario(args):
    if args.scenario not in SCENARIOS:
        raise SystemExit(f"Unknown scenario: {args.scenario}. Available: {list(SCENARIOS.keys())}")

    sc = SCENARIOS[args.scenario]
    client = OllamaClient(base_url=args.ollama_url)

    interviewer = InterviewerAgent(client, model=args.interviewer_model)
    observer = ObserverAgent(client, model=args.observer_model)
    hiring_manager = HiringManagerAgent(client, model=args.manager_model)

    engine = InterviewEngine(
        position=sc["position"],
        target_grade=sc["grade"],
        experience=sc["experience"],
        participant_name=sc["participant_name"],
        interviewer=interviewer,
        observer=observer,
        hiring_manager=hiring_manager,
    )

    print("\n=== Scenario run ===")
    print("BOT>", engine.get_last_visible_message(), "\n")

    for i, user_msg in enumerate(sc["user_messages"], start=1):
        print(f"YOU({i})> {user_msg}")
        # если сценарий явно просит фидбэк — завершаем
        if "фидбэк" in user_msg.lower():
            break
        bot_msg = engine.step(user_msg)
        print(f"BOT> {bot_msg}\n")

    feedback = engine.finish()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"logs/{args.scenario}_{ts}.json"
    engine.logger.save(out_path)

    print("\n=== FINAL FEEDBACK saved ===")
    print("Saved log to:", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--interviewer-model", default="qwen2.5vl:latest")
    p.add_argument("--observer-model", default="qwen2.5vl:latest")
    p.add_argument("--manager-model", default="qwen2.5vl:latest")

    sub = p.add_subparsers(dest="mode", required=True)

    intr = sub.add_parser("interactive")
    intr.add_argument("--name", default="Кандидат")
    intr.add_argument("--position", default="Backend Developer")
    intr.add_argument("--grade", default="Junior")
    intr.add_argument("--experience", default="Pet-проекты, базовый Python.")

    sc = sub.add_parser("scenario")
    sc.add_argument("--scenario", default="secret_example")

    args = p.parse_args()

    if args.mode == "interactive":
        run_interactive(args)
    else:
        run_scenario(args)


if __name__ == "__main__":
    main()
