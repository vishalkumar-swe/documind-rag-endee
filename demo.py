#!/usr/bin/env python3
"""
DocuMind CLI Demo
Quickly ingests sample documents and runs Q&A entirely from the command line.
Usage:
    python demo.py
    python demo.py --question "What is the capital of France?"
"""

import argparse
import json
from src.rag_engine import RAGEngine
from src.qa_pipeline import QAPipeline

# ── Sample documents ──────────────────────────
SAMPLE_DOCS = [
    (
        "climate_change.txt",
        """
        Climate change refers to long-term shifts in temperatures and weather patterns.
        Since the 1800s, human activities have been the main driver of climate change,
        primarily due to burning fossil fuels like coal, oil and gas. This produces
        heat-trapping gases. The effects include rising sea levels, more intense storms,
        droughts, and heatwaves. The Paris Agreement of 2015 committed nations to limit
        global warming to 1.5°C above pre-industrial levels. Renewable energy sources
        such as solar and wind power are central to reducing carbon emissions.
        Electric vehicles, energy-efficient buildings, and reforestation are also key
        strategies in the fight against climate change.
        """,
    ),
    (
        "machine_learning.txt",
        """
        Machine learning (ML) is a subset of artificial intelligence that enables systems
        to learn and improve from experience without being explicitly programmed.
        Supervised learning uses labelled data to train models for classification and
        regression tasks. Unsupervised learning finds hidden patterns in unlabelled data.
        Reinforcement learning trains agents through rewards and penalties.
        Deep learning, a subset of ML, uses neural networks with many layers to learn
        representations from large datasets. Applications include image recognition,
        natural language processing, recommendation systems, and autonomous vehicles.
        Popular frameworks include TensorFlow, PyTorch, and scikit-learn.
        """,
    ),
    (
        "space_exploration.txt",
        """
        Space exploration began in earnest with the Soviet Union's Sputnik 1 satellite
        in 1957. NASA's Apollo program landed humans on the Moon in 1969. The International
        Space Station, a collaboration between NASA, ESA, Roscosmos, JAXA, and CSA,
        has been continuously inhabited since 2000. SpaceX revolutionised the industry
        with reusable rockets like the Falcon 9 and plans for Mars colonisation via
        the Starship programme. The James Webb Space Telescope, launched in 2021, provides
        unprecedented views of the early universe. Artemis missions aim to return humans
        to the Moon by 2026 as a stepping stone to Mars.
        """,
    ),
]

DEMO_QUESTIONS = [
    "What caused climate change?",
    "How does supervised learning work?",
    "When did humans first land on the Moon?",
    "What is the Paris Agreement?",
    "What frameworks are used in deep learning?",
]


def run_demo(custom_question: str = None):
    print("\n" + "═" * 60)
    print("  DocuMind — RAG Demo (powered by Endee Vector DB)")
    print("═" * 60)

    # 1. Initialise
    print("\n[1/3] Initialising RAG engine …")
    rag = RAGEngine()
    qa  = QAPipeline(rag_engine=rag)

    # 2. Ingest sample documents
    print("\n[2/3] Ingesting sample documents …")
    for filename, text in SAMPLE_DOCS:
        result = rag.ingest_text(text, filename=filename)
        print(f"  ✓ {result['filename']}  ({result['num_chunks']} chunks, doc_id={result['doc_id']})")

    # 3. Ask questions
    questions = [custom_question] if custom_question else DEMO_QUESTIONS
    print(f"\n[3/3] Running {len(questions)} Q&A queries …\n")

    for q in questions:
        print("─" * 60)
        print(f"Q: {q}")
        result = qa.ask(q, top_k=3)
        print(f"A: {result['answer']}")
        print(f"\nMode: {result['mode']}")
        print("Sources:")
        for s in result["sources"]:
            print(f"  • {s['filename']}  (sim={s['similarity']})  {s['excerpt'][:80]}…")
        print()

    print("═" * 60)
    print("  Demo complete!")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocuMind CLI demo")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Ask a custom question instead of the defaults")
    args = parser.parse_args()
    run_demo(custom_question=args.question)
