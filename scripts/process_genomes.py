"""
process_genomes.py
------------------
Recursively scans data/harmful/ and data/non_harmful/ for .fna FASTA files,
extracts the following features from each genome, and saves the labeled dataset
to data/genome_dataset.csv.

Features extracted per genome
──────────────────────────────
  gc_content      – (G + C) / total_clean_bases
  genome_length   – total clean-base count (A, C, G, T only)
  shannon_entropy – H = -Σ p(b) log2 p(b)  over the four nucleotide bases
  <64 3-mer cols> – normalized trinucleotide (3-mer) frequencies
  <256 4-mer cols>– normalized tetranucleotide (4-mer) frequencies

Labels
──────
  harmful     → 1
  non_harmful → 0
"""

import os
import math
import itertools
from pathlib import Path

import pandas as pd
from Bio import SeqIO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
DATA_DIR = BASE_DIR / "data"

SOURCES = {
    DATA_DIR / "harmful":     1,   # label = 1
    DATA_DIR / "non_harmful": 0,   # label = 0
}

VALID_BASES = set("ACGT")

# Pre-compute all possible k-mers once at import time
ALL_3MERS = ["".join(p) for p in itertools.product("ACGT", repeat=3)]   # 64
ALL_4MERS = ["".join(p) for p in itertools.product("ACGT", repeat=4)]   # 256

# Column order for the final CSV (metadata → scalar features → k-mer blocks)
META_COLS    = ["filename", "label"]
SCALAR_COLS  = ["gc_content", "genome_length", "shannon_entropy"]
FEATURE_COLS = SCALAR_COLS + ALL_3MERS + ALL_4MERS
ALL_COLS     = META_COLS + FEATURE_COLS

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clean_sequence(seq: str) -> str:
    """Return only valid DNA characters (A, C, G, T) in uppercase."""
    return "".join(base for base in seq.upper() if base in VALID_BASES)


def compute_gc_content(sequence: str) -> float:
    """Fraction of G and C bases in the cleaned sequence."""
    if not sequence:
        return 0.0
    gc = sum(1 for b in sequence if b in ("G", "C"))
    return gc / len(sequence)


def compute_shannon_entropy(sequence: str) -> float:
    """
    Shannon entropy (bits) over the four nucleotide bases:
        H = -Σ p(b) * log2(p(b))  for b in {A, C, G, T}
    Maximum value is 2.0 bits (uniform distribution).
    Returns 0.0 for an empty sequence.
    """
    n = len(sequence)
    if n == 0:
        return 0.0
    entropy = 0.0
    for base in "ACGT":
        count = sequence.count(base)
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)
    return entropy


def compute_kmer_frequencies(sequence: str, all_kmers: list, k: int) -> dict:
    """
    Slide a window of length k across the sequence, count each k-mer,
    and return normalized frequencies (counts / total k-mers).
    All possible k-mers are initialised to 0 so the shape is always fixed.
    """
    counts = {kmer: 0 for kmer in all_kmers}
    total  = 0
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i: i + k]
        if kmer in counts:
            counts[kmer] += 1
            total += 1
    if total > 0:
        counts = {kmer: cnt / total for kmer, cnt in counts.items()}
    return counts


def process_fna_file(filepath: Path) -> dict | None:
    """
    Parse a FASTA file, concatenate all records into one sequence, clean it,
    and return a feature dictionary containing:
        - gc_content
        - genome_length
        - shannon_entropy
        - 64 normalised 3-mer frequencies
        - 256 normalised 4-mer frequencies

    Returns None if the file produces an empty sequence after cleaning.
    """
    raw_seq = ""
    try:
        for record in SeqIO.parse(str(filepath), "fasta"):
            raw_seq += str(record.seq)
    except Exception as exc:
        print(f"  [WARNING] Could not parse {filepath.name}: {exc}")
        return None

    cleaned = clean_sequence(raw_seq)
    if not cleaned:
        print(f"  [WARNING] No valid DNA bases found in {filepath.name}. Skipping.")
        return None

    features: dict = {}

    # ── Scalar genomic features ──────────────────────────────────────────────
    features["gc_content"]      = compute_gc_content(cleaned)
    features["genome_length"]   = len(cleaned)
    features["shannon_entropy"] = compute_shannon_entropy(cleaned)

    # ── 3-mer frequencies ────────────────────────────────────────────────────
    features.update(compute_kmer_frequencies(cleaned, ALL_3MERS, k=3))

    # ── 4-mer frequencies ────────────────────────────────────────────────────
    features.update(compute_kmer_frequencies(cleaned, ALL_4MERS, k=4))

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    rows = []

    for source_dir, label in SOURCES.items():
        if not source_dir.exists():
            print(f"[WARNING] Directory not found, skipping: {source_dir}")
            continue

        fna_files = sorted(source_dir.rglob("*.fna"))
        print(
            f"\n[INFO] Found {len(fna_files)} .fna file(s) in "
            f"{source_dir.relative_to(BASE_DIR)}"
        )

        for fna_path in fna_files:
            print(f"  Processing: {fna_path.name}")
            feature_dict = process_fna_file(fna_path)
            if feature_dict is None:
                continue

            row = {
                "filename": fna_path.name,
                "label":    label,
                **feature_dict,
            }
            rows.append(row)

    if not rows:
        print("\n[ERROR] No valid genome files were processed. Dataset is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=ALL_COLS)
    df.fillna(0, inplace=True)
    return df


def main():
    print("=" * 60)
    print("  Pathogen Classification — Genome Feature Extraction")
    print("=" * 60)

    df = build_dataset()

    if df.empty:
        print("\n[ABORTED] No data to save.")
        return

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "genome_dataset.csv"
    df.to_csv(output_path, index=False)

    n_3mer  = len(ALL_3MERS)
    n_4mer  = len(ALL_4MERS)
    n_total = df.shape[1]

    print(f"\n[SUCCESS] Dataset saved to: {output_path}")
    print(f"[INFO]    Shape : {df.shape[0]} rows x {n_total} columns")
    print(f"          +-- metadata        : {len(META_COLS)}   (filename, label)")
    print(f"          +-- scalar features : {len(SCALAR_COLS)}   (gc_content, genome_length, shannon_entropy)")
    print(f"          +-- 3-mer features  : {n_3mer} (normalized trinucleotide freqs)")
    print(f"          +-- 4-mer features  : {n_4mer} (normalized tetranucleotide freqs)")


if __name__ == "__main__":
    main()
