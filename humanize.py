"""Execution-footprint humanization (LIVE ONLY).

Makes live order flow look less robotic -- like a discretionary trader rather
than a metronome. This has NO effect on strategy signals and NO effect on the
backtest; it only jitters *timing* and varies cosmetic parts of order comments.

Position SIZE is intentionally left fully deterministic -- size randomization
was deliberately excluded.

Everything is gated by HUMANIZE_EXECUTION so it can be turned off instantly.
"""
import random
import time
import config


def entry_delay():
    """Brief random hesitation before firing an order (seconds). Returns the
    delay actually slept."""
    if not getattr(config, 'HUMANIZE_EXECUTION', False):
        return 0.0
    lo = float(getattr(config, 'HUMANIZE_ENTRY_DELAY_MIN', 0.4))
    hi = float(getattr(config, 'HUMANIZE_ENTRY_DELAY_MAX', 2.5))
    if hi <= lo:
        d = max(0.0, lo)
    else:
        d = random.uniform(lo, hi)
    if d > 0:
        try:
            time.sleep(d)
        except Exception:
            pass
    return d


def scan_wait(base_seconds):
    """Jitter a polling interval by +/- HUMANIZE_SCAN_JITTER_FRAC so the scan
    heartbeat isn't perfectly regular. Never returns < 0.2s."""
    if not getattr(config, 'HUMANIZE_EXECUTION', False):
        return base_seconds
    frac = float(getattr(config, 'HUMANIZE_SCAN_JITTER_FRAC', 0.5))
    frac = max(0.0, min(0.9, frac))
    return max(0.2, base_seconds * (1.0 + random.uniform(-frac, frac)))


_PREFIXES = ["ICT", "ict", "Ict", "SMC", "smc"]
_SEPS = [" ", "-", "_", " | ", ": "]


def rotate_comment(base_comment):
    """Vary an order comment slightly so the footprint isn't identical.

    IMPORTANT: method keywords (FVG / SILVER / SB / MMXM / 2022 / 2025 / OB /
    RETEST) are preserved verbatim -- position recovery keys off the magic
    number first, and get_method_from_position/deal also scan the comment, so we
    only reshuffle the cosmetic leading prefix / separator and maybe add a short
    numeric suffix. We never strip method tokens.
    """
    if not getattr(config, 'HUMANIZE_EXECUTION', False) or \
            not getattr(config, 'HUMANIZE_COMMENT_ROTATE', True):
        return base_comment
    c = base_comment
    if c.startswith("ICT "):
        c = random.choice(_PREFIXES) + random.choice(_SEPS) + c[4:]
    if random.random() < 0.5:
        c = c + random.choice(["", "", " " + str(random.randint(1, 99))])
    return c[:31]  # MT5 order comment length guard
