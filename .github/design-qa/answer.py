#!/usr/bin/env python3
"""Answer a /design-qa comment on a pull request with Claude, and post the reply.

.github/workflows/design-qa.yaml runs this. It calls the Messages API directly
with the read-only tools defined below (read a file, list files, grep, git
log/show/blame, gh issue and pull request reads), not through Claude Code.
instructions.md next to this file is the system prompt.

Local test, which prints the answer instead of posting it:

  ANTHROPIC_API_KEY=... python3 answer.py --dry-run --pr 4284 \\
      --charm <checkout of the PR branch> --lci <LCI checkout> \\
      --notes <charm-notes checkout> --question "..."
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
EFFORT = "high"
MAX_TOKENS = 32000
MAX_TURNS = 40
COST_LIMIT_USD = 4.0
# USD per million tokens for claude-opus-5-5; cache writes use the 5-minute TTL.
PRICE = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0}
MAX_RESULT_CHARS = 30000

HERE = Path(__file__).resolve().parent
ROOTS: dict[str, Path] = {}
REPO = "charmplusplus/charm"

ROOT_PROP = {
    "type": "string",
    "enum": ["charm", "lci", "charm-notes"],
    "description": "charm: the pull request's branch, with the reconverse "
    "submodule at contrib/reconverse. lci: the LCI sources reconverse pins. "
    "charm-notes: the practitioner notes.",
}

TOOLS = [
    {
        "name": "read_file",
        "description": "Read lines of a text file. Returns them numbered.",
        "input_schema": {
            "type": "object",
            "properties": {
                "root": ROOT_PROP,
                "path": {"type": "string", "description": "Path relative to the root."},
                "start_line": {"type": "integer", "description": "First line, from 1. Default 1."},
                "num_lines": {"type": "integer", "description": "Number of lines, at most 2000. Default 600."},
            },
            "required": ["root", "path"],
        },
    },
    {
        "name": "list_files",
        "description": "List the tracked files under a directory (recursively), optionally filtered by a glob on the file name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "root": ROOT_PROP,
                "path": {"type": "string", "description": "Directory relative to the root. Default the root."},
                "name_glob": {"type": "string", "description": "For example '*.md' or 'ckrdma*'."},
            },
            "required": ["root"],
        },
    },
    {
        "name": "grep",
        "description": "Search tracked files with git grep (extended regular expressions). Returns path:line:text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "root": ROOT_PROP,
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "A directory or file relative to the root, or a pathspec such as 'src/*.C' (* matches across directories). Default the root."},
                "ignore_case": {"type": "boolean"},
                "context": {"type": "integer", "description": "Lines of context around each match, at most 10. Default 0."},
            },
            "required": ["root", "pattern"],
        },
    },
    {
        "name": "git",
        "description": "Run git log, git show or git blame. The history is shallow (about 50 commits on the charm root).",
        "input_schema": {
            "type": "object",
            "properties": {
                "root": ROOT_PROP,
                "subdir": {"type": "string", "description": "Run in this subdirectory of the root, for example contrib/reconverse for the reconverse submodule's history. Default the root."},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Arguments starting with the subcommand, for example [\"log\", \"--oneline\", \"-5\", \"--\", \"src/ck-core/ckrdma.C\"]."},
            },
            "required": ["root", "args"],
        },
    },
    {
        "name": "github",
        "description": "Run a read-only gh command: issue view, issue list, pr view, pr list, pr diff, search issues, search prs. "
        "The default repository is charmplusplus/charm; pass -R charmplusplus/reconverse or -R uiuc-hpc/lci for the others.",
        "input_schema": {
            "type": "object",
            "properties": {
                "args": {"type": "array", "items": {"type": "string"}, "description": "Arguments after gh, for example [\"issue\", \"view\", \"4285\", \"--comments\"]."},
            },
            "required": ["args"],
        },
    },
]

GH_ALLOWED = {("issue", "view"), ("issue", "list"), ("pr", "view"), ("pr", "list"),
              ("pr", "diff"), ("search", "issues"), ("search", "prs")}
GIT_ALLOWED = {"log", "show", "blame"}


class ToolError(Exception):
    pass


def cap(text: str) -> str:
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + f"\n[truncated: {len(text) - MAX_RESULT_CHARS} more characters]"
    return text


def resolve(root: str, path: str = ".") -> Path:
    if root not in ROOTS:
        raise ToolError(f"unknown root {root!r}")
    base = ROOTS[root]
    target = (base / (path or ".")).resolve()
    if target != base and base not in target.parents:
        raise ToolError("path is outside the root")
    return target


def run(cmd: list[str], cwd: Path, ok_codes=(0,)) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", timeout=120)
    if p.returncode not in ok_codes:
        raise ToolError(f"exit {p.returncode}: {p.stderr.strip()[:2000]}")
    return p.stdout


def tool_read_file(root, path, start_line=1, num_lines=600):
    f = resolve(root, path)
    if f.is_dir():
        raise ToolError("that is a directory; use list_files")
    if not f.is_file():
        raise ToolError("no such file")
    start = max(1, int(start_line))
    count = min(max(1, int(num_lines)), 2000)
    lines = f.read_text(errors="replace").splitlines()
    chunk = lines[start - 1:start - 1 + count]
    out = "\n".join(f"{i:6}\t{line}" for i, line in enumerate(chunk, start))
    return out + f"\n[{len(lines)} lines in the file]"


def tool_list_files(root, path=".", name_glob=None):
    d = resolve(root, path)
    rel = os.path.relpath(d, ROOTS[root])
    files = run(["git", "ls-files", "--recurse-submodules", "--", rel], ROOTS[root]).splitlines()
    if name_glob:
        files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), name_glob)]
    if len(files) > 800:
        return "\n".join(files[:800]) + f"\n[{len(files) - 800} more; narrow the path or glob]"
    return "\n".join(files) or "No files."


def tool_grep(root, pattern, path=".", ignore_case=False, context=0):
    base = ROOTS.get(root)
    if base is None:
        raise ToolError(f"unknown root {root!r}")
    if "*" not in (path or "."):
        resolve(root, path)  # reject paths that escape the root
    elif ".." in Path(path).parts or path.startswith("/"):
        raise ToolError("path is outside the root")
    cmd = ["git", "grep", "--recurse-submodules", "-n", "-I", "-E"]
    if ignore_case:
        cmd.append("-i")
    ctx = min(max(0, int(context or 0)), 10)
    if ctx:
        cmd.append(f"-C{ctx}")
    cmd += ["-e", pattern, "--", path or "."]
    out = run(cmd, base, ok_codes=(0, 1))
    return out or "No matches."


def tool_git(root, args, subdir="."):
    cwd = resolve(root, subdir)
    if not args or args[0] not in GIT_ALLOWED:
        raise ToolError(f"allowed subcommands: {sorted(GIT_ALLOWED)}")
    for a in args:
        if a.startswith("--output") or a.startswith("--ext-diff"):
            raise ToolError(f"option not allowed: {a}")
    return run(["git", "--no-pager", *args], cwd) or "(no output)"


def tool_github(args):
    if len(args) < 2 or (args[0], args[1]) not in GH_ALLOWED:
        raise ToolError(f"allowed: {sorted(' '.join(p) for p in GH_ALLOWED)}")
    if any(a in ("--web", "-w") for a in args):
        raise ToolError("--web is not allowed")
    return run(["gh", *args], ROOTS["charm"]) or "(no output)"


TOOL_FUNCS = {"read_file": tool_read_file, "list_files": tool_list_files,
              "grep": tool_grep, "git": tool_git, "github": tool_github}


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Returns (result text, is_error)."""
    func = TOOL_FUNCS.get(name)
    if func is None:
        return f"unknown tool {name}", True
    if not isinstance(args, dict):
        return "tool input must be an object", True
    try:
        return cap(func(**args)), False
    except ToolError as e:
        return str(e), True
    except TypeError as e:
        return f"bad arguments: {e}", True
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"failed: {e}", True


# --- GitHub ------------------------------------------------------------------

def gh_api(method: str, path: str, payload=None):
    cmd = ["gh", "api", "-X", method, path]
    tmp = None
    if payload is not None:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, tmp)
        tmp.close()
        cmd += ["--input", tmp.name]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    finally:
        if tmp:
            os.unlink(tmp.name)
    if p.returncode != 0:
        raise RuntimeError(f"gh api {method} {path}: {p.stderr.strip()}")
    return json.loads(p.stdout) if p.stdout.strip() else None


def defuse_mentions(text: str) -> str:
    """Put a zero-width space after @ outside code, so the reply notifies nobody."""
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<![\w`])@(?=[A-Za-z0-9])", "@​", parts[i])
    return "".join(parts)


def pr_context(pr: int) -> tuple[str, list[str]]:
    p = subprocess.run(["gh", "pr", "view", str(pr), "-R", REPO, "--json", "title,files"],
                       capture_output=True, text=True, timeout=120, check=True)
    info = json.loads(p.stdout)
    docs = [f["path"] for f in info.get("files", [])
            if f["path"].startswith("doc/") and f["path"].endswith(".md")]
    return info.get("title", ""), docs


def review_thread(pr: int, comment: dict) -> list[dict]:
    """Earlier comments in the review thread of a comment on a line of the diff."""
    root_id = comment.get("in_reply_to_id") or comment["id"]
    p = subprocess.run(["gh", "api", "--paginate", f"repos/{REPO}/pulls/{pr}/comments?per_page=100",
                        "--jq", ".[] | {id, in_reply_to_id, created_at, body, user: {login: .user.login}}"],
                       capture_output=True, text=True, timeout=120, check=True)
    comments = [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    thread = [c for c in comments
              if (c["id"] == root_id or c.get("in_reply_to_id") == root_id) and c["id"] != comment["id"]]
    thread.sort(key=lambda c: c["created_at"])
    return thread[-10:]


# --- The model ------------------------------------------------------------------

def cost_of(u) -> float:
    return (u["input"] * PRICE["input"] + u["output"] * PRICE["output"]
            + u["cache_read"] * PRICE["cache_read"] + u["cache_write"] * PRICE["cache_write"]) / 1e6


def ask(question_block: str) -> tuple[str, dict]:
    client = anthropic.Anthropic()
    system = (HERE / "instructions.md").read_text()
    messages = [{"role": "user", "content": question_block}]
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "turns": 0}
    stop_asked = False
    for turn in range(1, MAX_TURNS + 1):
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},
        ) as stream:
            resp = stream.get_final_message()
        u = resp.usage
        usage["input"] += u.input_tokens or 0
        usage["output"] += u.output_tokens or 0
        usage["cache_read"] += u.cache_read_input_tokens or 0
        usage["cache_write"] += u.cache_creation_input_tokens or 0
        usage["turns"] = turn
        spent = cost_of(usage)
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        print(f"turn {turn}: stop={resp.stop_reason} tools={len(tool_uses)} "
              f"in={u.input_tokens} cached={u.cache_read_input_tokens} out={u.output_tokens} "
              f"total=${spent:.2f}", flush=True)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("the model's reply hit max_tokens")
        if resp.stop_reason == "refusal" or not tool_uses:
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if not text:
                raise RuntimeError(f"no answer text (stop reason {resp.stop_reason})")
            usage["cost"] = spent
            return text, usage
        if stop_asked:
            raise RuntimeError("the model kept calling tools after being asked to answer")
        last_chance = spent > COST_LIMIT_USD or turn >= MAX_TURNS - 1
        results = []
        for b in tool_uses:
            if last_chance:
                text, err = ("Not run: the budget for this question is used up. "
                             "Answer now with what you have, and say what you could not check."), True
            else:
                print(f"  {b.name} {json.dumps(b.input)[:300]}", flush=True)
                text, err = run_tool(b.name, b.input)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": err})
        stop_asked = last_chance
        messages.append({"role": "user", "content": results})
    raise RuntimeError("ran out of turns")


def build_question(pr: int, title: str, docs: list[str], author: str, body: str,
                   review: dict | None, thread: list[dict]) -> str:
    lines = [f"Pull request #{pr}: {title}",
             "Design document(s) in this pull request: " + (", ".join(docs) or "none found under doc/"),
             ""]
    if review:
        lines += [f"{author} commented on a line of the diff: {review.get('path')}, "
                  f"line {review.get('line') or review.get('original_line')}.",
                  "The diff hunk the comment is attached to:", "```diff", review.get("diff_hunk", ""), "```"]
        if thread:
            lines.append("Earlier comments in this thread, oldest first:")
            for c in thread:
                lines += [f"--- {c['user']['login']}:", c["body"][:3000]]
            lines.append("---")
    else:
        lines.append(f"{author} commented on the pull request's conversation tab. If the comment refers "
                     f"to earlier comments, read them with the github tool: pr view {pr} --comments.")
    lines += ["", "The comment:", "<comment>", body, "</comment>"]
    return "\n".join(lines)


def main():
    global REPO
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the answer; do not post")
    ap.add_argument("--pr", type=int)
    ap.add_argument("--question")
    ap.add_argument("--charm", default=os.environ.get("CHARM_DIR"))
    ap.add_argument("--lci", default=os.environ.get("LCI_DIR"))
    ap.add_argument("--notes", default=os.environ.get("NOTES_DIR"))
    a = ap.parse_args()
    for name, d in (("charm", a.charm), ("lci", a.lci), ("charm-notes", a.notes)):
        if not d or not Path(d).is_dir():
            sys.exit(f"directory for root {name} not given or missing: {d}")
        ROOTS[name] = Path(d).resolve()

    review = None
    thread: list[dict] = []
    if a.dry_run:
        if not (a.pr and a.question):
            sys.exit("--dry-run needs --pr and --question")
        pr, body, author = a.pr, a.question, "the person asking"
    else:
        REPO = os.environ["GITHUB_REPOSITORY"]
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        comment = event["comment"]
        body, author = comment["body"], comment["user"]["login"]
        if os.environ["GITHUB_EVENT_NAME"] == "pull_request_review_comment":
            pr = event["pull_request"]["number"]
            review = comment
            kind = "pulls"
        else:
            pr = event["issue"]["number"]
            kind = "issues"
        gh_api("POST", f"repos/{REPO}/{kind}/comments/{comment['id']}/reactions", {"content": "eyes"})
        if review:
            thread = review_thread(pr, comment)
    os.environ["GH_REPO"] = REPO

    run_id = os.environ.get("GITHUB_RUN_ID")
    run_link = (f"[Run log]({os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{REPO}/actions/runs/{run_id})"
                if run_id else "Local run")
    try:
        title, docs = pr_context(pr)
        answer, usage = ask(build_question(pr, title, docs, author, body, review, thread))
        reply = (defuse_mentions(answer)
                 + f"\n\n<sub>design-qa bot ({MODEL}). This answer cost about ${usage['cost']:.2f}, "
                 f"paid from a budget set aside for these questions. {run_link}</sub>")
        print(f"done: {usage['turns']} turns, ${usage['cost']:.2f}", flush=True)
    except anthropic.APIStatusError as e:
        reply = (f"The design-qa bot could not answer: API error {e.status_code}: {e.message} "
                 f"(request {e.request_id}). {run_link}")
        print(reply, file=sys.stderr)
        if not a.dry_run:
            post(review, pr, reply)
        sys.exit(1)
    except Exception as e:
        reply = f"The design-qa bot failed: {type(e).__name__}: {e}. {run_link}"
        print(reply, file=sys.stderr)
        if not a.dry_run:
            post(review, pr, reply)
        raise

    if a.dry_run:
        print("\n" + reply)
    else:
        post(review, pr, reply)


def post(review: dict | None, pr: int, body: str):
    if review:
        gh_api("POST", f"repos/{REPO}/pulls/{pr}/comments/{review['id']}/replies", {"body": body})
    else:
        gh_api("POST", f"repos/{REPO}/issues/{pr}/comments", {"body": body})


if __name__ == "__main__":
    main()
