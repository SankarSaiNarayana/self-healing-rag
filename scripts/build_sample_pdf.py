#!/usr/bin/env python3
"""Build a formatted sample PDF under data/ (for demos and ingest)."""
from __future__ import annotations

import argparse
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos


class PDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)
        # Core fonts only support Latin-1; avoid Unicode dashes/quotes.
        self.cell(0, 10, "Self-healing RAG - Sample document", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def build_pdf(out_path: Path) -> None:
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 11)

    sections = [
        (
            "Overview",
            "This sample describes a self-healing Retrieval-Augmented Generation (RAG) service. "
            "It goes beyond a single retrieve-then-generate step by adding quality checks, "
            "verification loops, and optional memory so answers stay grounded in your documents.",
        ),
        (
            "Pipeline (high level)",
            "1) Classify the user question (factual, conceptual, or multi-hop).\n"
            "2) Hybrid retrieval: dense vectors plus BM25, with optional re-ranking.\n"
            "3) Check retrieval quality; if weak, broaden and re-query.\n"
            "4) Generate an answer using only retrieved context.\n"
            "5) Verify claims against sources; retry retrieval/generation when needed.\n"
            "6) Return the answer with citations and a confidence score.",
        ),
        (
            "What to expect",
            "Upload your own PDFs or text under the data directory, run ingest, then query the API. "
            "For production, configure a vector database (e.g. Qdrant) and an OpenAI-compatible LLM endpoint.",
        ),
    ]

    for title, body in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, title)
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/demo.pdf"))
    args = p.parse_args()
    build_pdf(args.out)


if __name__ == "__main__":
    main()
