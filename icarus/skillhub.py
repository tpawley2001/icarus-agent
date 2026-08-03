"""Install skills from reputable public repositories.

Icarus' runtime is fully local — this module is the one deliberate exception,
and it only ever runs when *you* type `/skills install`. It talks to github.com
and nowhere else, and by default only to a small allowlist of first-party
repositories (the same set hermes treats as trusted):

    anthropics/skills   openai/skills   huggingface/skills   NVIDIA/skills

A skill is executable instructions handed to an agent that can run shell
commands, so anything fetched is scanned before it lands on disk and anything
that looks like remote-code-execution or credential access is surfaced for you
to approve. Provenance (repo, path, blob sha, timestamp) is recorded in
~/.icarus/skills.lock.json so `/skills update` can tell what changed.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import config

TRUSTED_REPOS: Tuple[str, ...] = (
    "anthropics/skills",
    "openai/skills",
    "huggingface/skills",
    "NVIDIA/skills",
)

INDEX_CACHE = config.HOME / "hub_index.json"
LOCKFILE = config.HOME / "skills.lock.json"
INDEX_TTL = 24 * 3600
UA = "icarus-skillhub/1.0 (+local agent)"

# Files worth pulling alongside SKILL.md. A skill's supporting material is
# usually reference text and scripts; binaries and archives are not fetched.
COMPANION_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".sh", ".toml", ".csv"}
MAX_FILE_BYTES = 512_000
MAX_FILES_PER_SKILL = 40

# Patterns worth stopping on. Tuned for the real risk in an instructions file:
# telling an agent to pipe the network into a shell, or to read secrets.
RISK_PATTERNS: List[Tuple[str, str, str]] = [
    ("critical", r"curl[^\n|]*\|\s*(ba|z|k)?sh", "pipes a download straight into a shell"),
    ("critical", r"wget[^\n|]*\|\s*(ba|z|k)?sh", "pipes a download straight into a shell"),
    ("critical", r"base64\s+(-d|--decode)[^\n|]*\|\s*(ba|z|k)?sh", "executes base64-decoded code"),
    ("critical", r"\beval\s*\(\s*(base64|atob|requests\.get|urlopen)", "evaluates fetched code"),
    ("high", r"\brm\s+-[rf]{1,2}\s+[/~]", "recursive delete of an absolute path"),
    ("high", r"(ssh|aws|gcloud|kube)[/\\]?(config|credentials)\b", "reads credential files"),
    ("high", r"\.env\b[^\n]{0,40}(cat|read|upload|post|curl)", "reads .env"),
    ("high", r"(API_KEY|SECRET|TOKEN|PASSWORD)\s*=\s*['\"][A-Za-z0-9/+_-]{16,}", "embedded credential"),
    ("high", r"id_rsa|id_ed25519|\.ssh/", "touches SSH keys"),
    ("medium", r"\bsudo\b", "requires root"),
    ("medium", r"\bchmod\s+777\b", "world-writable permissions"),
    ("medium", r"crontab\s+-|systemctl\s+(enable|start)", "installs persistence"),
]


class HubError(RuntimeError):
    pass


@dataclass
class RemoteSkill:
    name: str
    repo: str
    dir_path: str          # e.g. "skills/pdf"
    skill_path: str        # e.g. "skills/pdf/SKILL.md"
    description: str = ""
    sha: str = ""
    files: List[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.repo}:{self.name}"


@dataclass
class Finding:
    severity: str
    reason: str
    file: str
    line: int
    excerpt: str


def _get(url: str, timeout: int = 30, raw: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise HubError(
                "GitHub rate-limited this machine (60 requests/hour unauthenticated). "
                "Wait, or set GITHUB_TOKEN."
            ) from e
        raise HubError(f"HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise HubError(f"cannot reach {urllib.parse.urlparse(url).netloc}: {e.reason}") from e
    return body if raw else body.decode("utf-8", "replace")


def _api(url: str) -> Any:
    import os

    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"}
    )
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise HubError(
                "GitHub rate-limited this machine (60 req/hour unauthenticated). "
                "Set GITHUB_TOKEN to raise it."
            ) from e
        raise HubError(f"HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise HubError(f"cannot reach api.github.com: {e.reason}") from e


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def _index_repo(repo: str) -> List[RemoteSkill]:
    """One recursive tree call gives the whole repo layout."""
    meta = _api(f"https://api.github.com/repos/{repo}")
    branch = meta.get("default_branch", "main")
    tree = _api(
        f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    )
    entries = tree.get("tree", [])
    blobs = {t["path"]: t for t in entries if t.get("type") == "blob"}

    out: List[RemoteSkill] = []
    for path, node in blobs.items():
        if not path.endswith("SKILL.md"):
            continue
        dir_path = path[: -len("/SKILL.md")] if "/" in path else ""
        companions = [
            p for p in blobs
            if p.startswith(dir_path + "/")
            and p != path
            and Path(p).suffix.lower() in COMPANION_SUFFIXES
        ][:MAX_FILES_PER_SKILL]
        out.append(
            RemoteSkill(
                name=Path(dir_path).name or repo.split("/")[-1],
                repo=repo,
                dir_path=dir_path,
                skill_path=path,
                sha=node.get("sha", ""),
                files=companions,
            )
        )
    return out


def _describe(skills: List[RemoteSkill]) -> None:
    """Fill in descriptions from each SKILL.md's frontmatter (raw, no API quota)."""
    for s in skills:
        try:
            head = _get(_raw_url(s.repo, s.skill_path), timeout=20)[:4000]
        except HubError:
            continue
        m = re.match(r"^---\s*\n(.*?)\n---", head, re.DOTALL)
        if not m:
            continue
        try:
            meta = yaml.safe_load(m.group(1)) or {}
        except Exception:
            continue
        s.description = str(meta.get("description") or "").strip().replace("\n", " ")
        if meta.get("name"):
            s.name = str(meta["name"]).strip()


def _raw_url(repo: str, path: str, ref: str = "HEAD") -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{urllib.parse.quote(path)}"


def load_index(refresh: bool = False, repos: Optional[List[str]] = None) -> List[RemoteSkill]:
    repos = list(repos or TRUSTED_REPOS)
    if not refresh and INDEX_CACHE.exists():
        try:
            blob = json.loads(INDEX_CACHE.read_text())
            if time.time() - blob.get("fetched_at", 0) < INDEX_TTL:
                cached = [RemoteSkill(**s) for s in blob.get("skills", [])]
                if {s.repo for s in cached} >= set(repos):
                    return [s for s in cached if s.repo in repos]
        except Exception:
            pass

    found: List[RemoteSkill] = []
    errors: List[str] = []
    for repo in repos:
        try:
            batch = _index_repo(repo)
            _describe(batch)
            found.extend(batch)
        except HubError as e:
            errors.append(f"{repo}: {e}")
    if not found and errors:
        raise HubError("; ".join(errors))

    try:
        config.ensure_dirs()
        INDEX_CACHE.write_text(json.dumps(
            {"fetched_at": time.time(),
             "skills": [s.__dict__ for s in found]}, indent=1))
    except Exception:
        pass
    return found


def search(query: str, index: List[RemoteSkill]) -> List[RemoteSkill]:
    q = (query or "").lower().strip()
    if not q:
        return index
    scored = []
    for s in index:
        hay = f"{s.name} {s.description} {s.repo}".lower()
        if q in hay:
            # Exact name match first, then name-prefix, then anything.
            rank = 0 if s.name.lower() == q else (1 if q in s.name.lower() else 2)
            scored.append((rank, s.name, s))
    return [s for _, _, s in sorted(scored, key=lambda x: (x[0], x[1]))]


# ---------------------------------------------------------------------------
# safety scan
# ---------------------------------------------------------------------------

def scan(files: Dict[str, str]) -> List[Finding]:
    findings: List[Finding] = []
    compiled = [(sev, re.compile(pat, re.IGNORECASE), why) for sev, pat, why in RISK_PATTERNS]
    for path, text in files.items():
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > 2000:
                line = line[:2000]
            for sev, rx, why in compiled:
                if rx.search(line):
                    findings.append(Finding(sev, why, path, i, line.strip()[:160]))
    # One finding per (file, reason) is enough to make the decision.
    seen, unique = set(), []
    for f in findings:
        key = (f.file, f.reason)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    order = {"critical": 0, "high": 1, "medium": 2}
    return sorted(unique, key=lambda f: order.get(f.severity, 9))


def verdict(findings: List[Finding]) -> str:
    if any(f.severity == "critical" for f in findings):
        return "dangerous"
    if any(f.severity == "high" for f in findings):
        return "caution"
    return "safe"


# ---------------------------------------------------------------------------
# install / remove
# ---------------------------------------------------------------------------

def fetch(skill: RemoteSkill) -> Dict[str, str]:
    """Download a skill's files into memory so they can be scanned first."""
    files: Dict[str, str] = {}
    wanted = [skill.skill_path] + skill.files
    for path in wanted[: MAX_FILES_PER_SKILL + 1]:
        try:
            body = _get(_raw_url(skill.repo, path), raw=True)
        except HubError:
            continue
        if len(body) > MAX_FILE_BYTES:
            continue
        try:
            files[path] = body.decode("utf-8")
        except UnicodeDecodeError:
            continue  # skip binaries rather than write them blind
    if skill.skill_path not in files:
        raise HubError(f"could not download {skill.skill_path} from {skill.repo}")
    return files


def install(skill: RemoteSkill, files: Dict[str, str]) -> Path:
    """Write a fetched skill under ~/.icarus/skills/<owner>/<name>/."""
    owner = skill.repo.split("/")[0]
    dest = config.SKILLS_DIR / owner / skill.name
    dest.mkdir(parents=True, exist_ok=True)

    base = skill.dir_path
    for path, text in files.items():
        rel = path[len(base) + 1:] if base and path.startswith(base + "/") else Path(path).name
        target = dest / rel
        # Never let a repo path escape the skill directory.
        if not str(target.resolve()).startswith(str(dest.resolve())):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    lock = _read_lock()
    lock[skill.ref] = {
        "repo": skill.repo,
        "name": skill.name,
        "path": skill.dir_path,
        "sha": skill.sha,
        "installed_at": time.time(),
        "files": sorted(files),
        "dest": str(dest),
    }
    _write_lock(lock)
    return dest


def remove(ref_or_name: str) -> Optional[str]:
    import shutil

    lock = _read_lock()
    key = ref_or_name if ref_or_name in lock else next(
        (k for k, v in lock.items() if v.get("name") == ref_or_name), ""
    )
    if not key:
        return None
    dest = Path(lock[key].get("dest", ""))
    if dest.is_dir() and str(dest).startswith(str(config.SKILLS_DIR)):
        shutil.rmtree(dest, ignore_errors=True)
    del lock[key]
    _write_lock(lock)
    return key


def installed() -> Dict[str, Dict[str, Any]]:
    return _read_lock()


def outdated(index: List[RemoteSkill]) -> List[Tuple[str, str, str]]:
    """(ref, old_sha, new_sha) for installed skills whose upstream blob moved."""
    lock = _read_lock()
    by_ref = {s.ref: s for s in index}
    out = []
    for ref, meta in lock.items():
        remote = by_ref.get(ref)
        if remote and remote.sha and meta.get("sha") and remote.sha != meta["sha"]:
            out.append((ref, meta["sha"][:8], remote.sha[:8]))
    return out


def _read_lock() -> Dict[str, Dict[str, Any]]:
    if not LOCKFILE.exists():
        return {}
    try:
        return json.loads(LOCKFILE.read_text())
    except Exception:
        return {}


def _write_lock(lock: Dict[str, Dict[str, Any]]) -> None:
    try:
        config.ensure_dirs()
        LOCKFILE.write_text(json.dumps(lock, indent=1, sort_keys=True))
    except Exception:
        pass
