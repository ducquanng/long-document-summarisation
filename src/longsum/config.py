"""Study settings: model, sizes, and what counts as a long document."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
# The encoder cannot see past this many tokens, whatever you feed it.
MODEL_MAX_INPUT = 1024

SEED = 42
N_DOCUMENTS = 12
# Only documents that overflow the window: the whole question is what to do when
# truncation would throw text away.
MIN_DOCUMENT_TOKENS = 1200

SUMMARY_MIN_TOKENS = 40
SUMMARY_MAX_TOKENS = 120
