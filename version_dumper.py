"""Auto-bump BOT_VERSION in config.py based on file modifications.

Usage:
    python version_dumper.py           # Auto-bump patch if any files changed
    python version_dumper.py --minor   # Force a minor bump
    python version_dumper.py --major   # Force a major bump
    python version_dumper.py --dry-run # Preview changes without applying
"""
import hashlib
import json
import re
import argparse
import difflib
import shutil
from pathlib import Path

HERE = Path(__file__).parent
CONFIG = HERE / "config.py"
CACHE = HERE / "version_cache.json"
PY_GLOB = "*.py"

EXCLUDED = {"version_dumper.py", "version_cache.json", "dump.txt",
            "init_worker_dump.txt", "trade_memory.json", "fix_ver.py"}


def md5(path: Path) -> str:
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def count_lines(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = path.read_bytes().decode("utf-8", errors="replace")
    return len(text.splitlines())


def get_diff_lines(old_text: str, new_text: str) -> int:
    diff = difflib.ndiff(old_text.splitlines(), new_text.splitlines())
    changes = sum(1 for line in diff if line.startswith('+ ') or line.startswith('- '))
    return changes


def read_version() -> tuple[int, int, int]:
    text = CONFIG.read_text(encoding="utf-8")
    m = re.search(r'BOT_VERSION\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text)
    if not m:
        raise RuntimeError("BOT_VERSION not found in config.py")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def write_version(major: int, minor: int, patch: int) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    new_text = re.sub(
        r'BOT_VERSION\s*=\s*"\d+\.\d+\.\d+"',
        f'BOT_VERSION = "{major}.{minor}.{patch}"',
        text,
    )
    CONFIG.write_text(new_text, encoding="utf-8")


def load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": "0.0.0", "hashes": {}}


def save_cache(cache: dict) -> None:
    CACHE.write_text(json.dumps(cache, indent=4) + "\n", encoding="utf-8")


def check_and_bump_version(args=None, custom_logger=None) -> str | None:
    def log(msg):
        if custom_logger:
            custom_logger.info(f"[VERSION DUMPER] {msg}")
        else:
            print(msg)

    parser = argparse.ArgumentParser(description="Auto-bump BOT_VERSION based on file modifications.")
    parser.add_argument("--major", action="store_true", help="Force a major version bump")
    parser.add_argument("--minor", action="store_true", help="Force a minor version bump")
    parser.add_argument("--patch", action="store_true", help="Force a patch version bump")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without applying")
    parser.add_argument("--force", action="store_true", help="Force a bump even if no files changed")
    
    if args is not None:
        parsed_args, _ = parser.parse_known_args(args)
    else:
        parsed_args, _ = parser.parse_known_args()

    cache = load_cache()
    try:
        cur_major, cur_minor, cur_patch = read_version()
    except RuntimeError as e:
        log(f"Error: {e}")
        return

    old_hashes: dict[str, str] = cache.get("hashes", {})
    backup_dir = HERE / ".version_backup"
    py_files = sorted(HERE.glob(PY_GLOB))
    
    new_hashes: dict[str, str] = {}
    changed_files = []
    total_changed_lines = 0

    for fp in py_files:
        name = fp.name
        if name in EXCLUDED:
            new_hashes[name] = old_hashes.get(name, "")
            continue
            
        h = md5(fp)
        new_hashes[name] = h
        old_h = old_hashes.get(name)
        
        if old_h != h:
            changed_files.append(name)
            
            old_file = backup_dir / name
            try:
                new_text = fp.read_text(encoding="utf-8", errors="replace")
                if old_file.exists():
                    old_text = old_file.read_text(encoding="utf-8", errors="replace")
                    lines_changed = get_diff_lines(old_text, new_text)
                else:
                    lines_changed = count_lines(fp)
            except Exception:
                lines_changed = count_lines(fp)
                
            total_changed_lines += lines_changed

            if not old_h:
                log(f"  ADDED    {name} ({lines_changed} lines)")
            else:
                log(f"  MODIFIED {name} ({lines_changed} lines changed)")

    log(f"Current version: {cur_major}.{cur_minor}.{cur_patch}")

    if not changed_files and not (parsed_args.major or parsed_args.minor or parsed_args.patch or parsed_args.force):
        log("No changes detected. Run with --force to bump anyway.")
        return None

    new_major, new_minor, new_patch = cur_major, cur_minor, cur_patch
    bump_type = "PATCH"

    if parsed_args.major:
        new_major += 1
        new_minor = 0
        new_patch = 0
        bump_type = "MAJOR (Forced)"
    elif parsed_args.minor:
        new_minor += 1
        new_patch = 0
        bump_type = "MINOR (Forced)"
    elif parsed_args.patch:
        new_patch += 1
        bump_type = "PATCH (Forced)"
    else:
        # Auto-calculate based on line changes
        if total_changed_lines >= 150:
            new_major += 1
            new_minor = 0
            new_patch = 0
            bump_type = f"MAJOR ({total_changed_lines} lines)"
        elif total_changed_lines >= 50:
            new_minor += 1
            new_patch = 0
            bump_type = f"MINOR ({total_changed_lines} lines)"
        elif total_changed_lines >= 5:
            new_patch += 1
            bump_type = f"PATCH ({total_changed_lines} lines)"
        else:
            log(f"Only {total_changed_lines} lines changed (needs >= 5). No version bump.")
            return None

    log(f"Bump type:       {bump_type}")
    log(f"New version:     {new_major}.{new_minor}.{new_patch}")

    if parsed_args.dry_run:
        log("Dry-run complete. No changes applied.")
    else:
        write_version(new_major, new_minor, new_patch)
        
        # FIX: Recalculate config.py hash AFTER writing the new version to avoid infinite bump loops
        if "config.py" in new_hashes:
            new_hashes["config.py"] = md5(CONFIG)
            
        cache["version"] = f"{new_major}.{new_minor}.{new_patch}"
        cache["hashes"] = new_hashes
        save_cache(cache)
        
        # Save backups for future diffing
        backup_dir.mkdir(exist_ok=True)
        for fp in py_files:
            if fp.name not in EXCLUDED:
                shutil.copy2(fp, backup_dir / fp.name)
                
        log("Version updated successfully!")
        return f"{new_major}.{new_minor}.{new_patch}"

def main() -> None:
    check_and_bump_version()

if __name__ == "__main__":
    main()
