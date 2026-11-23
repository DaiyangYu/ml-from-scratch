"""drill：手写练习的复习调度工具。

    uv run drill                查看哪些题目到期
    uv run drill start [题目]   开始一次手写（把实现文件重置为空白骨架，并在 VSCode 中打开）
    uv run drill done  [题目]   写完：跑测试、评分、记录、提交，然后打开对比
    uv run drill check [题目]   只跑测试，不记录不提交
    uv run drill diff  [题目]   与上一次手写的版本对比；加 --ref 与参考实现对比
    uv run drill verify [题目]  检验测试本身：参考实现应通过，各种错误写法应被拦住

调度状态不单独存储：reviews.jsonl 是只追加的复习记录，
每次都由它重新推算出各题目的到期时间。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fsrs import Card, Rating, Scheduler

ROOT = Path(__file__).resolve().parents[2]
TOPICS = ROOT / "topics"
LOG = ROOT / "reviews.jsonl"
STATE = ROOT / ".drill"  # 本机临时状态，不进入 git
SESSION = STATE / "session.json"

# 期望记忆保留率：到期时仍能独立写出来的目标概率。调低则间隔更长、复习更少。
DESIRED_RETENTION = 0.9

# 间隔上限（天）。FSRS 的默认参数是用背卡片的数据拟合的，连续答对几次后间隔会
# 拉长到一年以上；手写整段代码比回忆一张卡片更容易生疏，所以设一个上限。
MAX_INTERVAL_DAYS = 90

# 手写一次要几十分钟，不适用背单词式的"几分钟后再来一遍"，所以去掉学习步骤，
# 间隔直接以天为单位；关闭随机扰动，保证由记录重新推算的结果始终一致。
SCHEDULER = Scheduler(
    desired_retention=DESIRED_RETENTION,
    learning_steps=(),
    relearning_steps=(),
    maximum_interval=MAX_INTERVAL_DAYS,
    enable_fuzzing=False,
)

RATINGS = {"again": Rating.Again, "hard": Rating.Hard, "good": Rating.Good, "easy": Rating.Easy}


# ---------- 基础工具 ----------

def now() -> datetime:
    override = os.environ.get("DRILL_NOW")  # 仅用于测试调度逻辑
    if override:
        return datetime.fromisoformat(override).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def die(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


def git(*args: str, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=capture, text=True)


def vscode(*args: str) -> bool:
    """用 VSCode 打开文件或对比视图。找不到 code 命令时返回 False。"""
    if os.environ.get("DRILL_NO_VSCODE") or shutil.which("code") is None:
        return False
    return subprocess.run(["code", *args], cwd=ROOT).returncode == 0


def all_topics() -> list[str]:
    if not TOPICS.is_dir():
        return []
    return sorted(p.name for p in TOPICS.iterdir() if (p / "stub.py").is_file())


def solution_path(topic: str) -> Path:
    return TOPICS / topic / f"{topic}.py"


def reference_path(topic: str) -> Path:
    return TOPICS / topic / f"{topic}_ref.py"


def load_reviews() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def card_of(topic: str, reviews: list[dict]) -> tuple[Card | None, list[dict]]:
    """由复习记录推算某题目当前的调度状态。"""
    history = [r for r in reviews if r["topic"] == topic]
    if not history:
        return None, history
    card = Card()
    for r in history:
        card, _ = SCHEDULER.review_card(card, RATINGS[r["rating"]], datetime.fromisoformat(r["time"]))
    return card, history


def days_until_due(card: Card) -> int:
    """按本地日期计算距离到期的天数，负数表示已逾期。"""
    return (card.due.astimezone().date() - now().astimezone().date()).days


def load_session() -> dict | None:
    if SESSION.exists():
        return json.loads(SESSION.read_text(encoding="utf-8"))
    return None


def save_session(session: dict) -> None:
    STATE.mkdir(exist_ok=True)
    SESSION.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")


def check_topic(topic: str) -> str:
    if topic not in all_topics():
        die(f"没有题目 {topic!r}。现有题目：{'、'.join(all_topics()) or '（无）'}")
    return topic


def resolve_topic(topic: str | None) -> str:
    """未指定题目时，取正在进行的那一次手写。"""
    if topic is None:
        session = load_session()
        if session is None:
            die("没有正在进行的手写，请指定题目，或先运行 drill start。")
        topic = session["topic"]
    return check_topic(topic)


# ---------- 查看到期 ----------

def due_rows() -> tuple[list, list, list]:
    """返回（到期、未开始、未到期）三组题目。"""
    reviews = load_reviews()
    due, fresh, later = [], [], []
    for topic in all_topics():
        card, history = card_of(topic, reviews)
        if card is None:
            fresh.append(topic)
            continue
        days = days_until_due(card)
        (due if days <= 0 else later).append((days, topic, card, history))
    return sorted(due), fresh, sorted(later)


def overdue_text(days: int) -> str:
    return "今天到期" if days == 0 else f"已逾期 {-days} 天"


def cmd_status(args: argparse.Namespace) -> None:
    topics = all_topics()
    due, fresh, later = due_rows()
    if getattr(args, "due_only", False):  # 供定时提醒使用：只输出到期的题目，没有则不输出
        for days, topic, _, _ in due:
            print(f"- {topic}（{overdue_text(days)}）")
        return
    if not topics:
        print("还没有题目。在 Claude Code 里说“新增题目 xxx”即可创建。")
        return

    width = max(len(t) for t in topics) + 2
    if due:
        print(f"到期（{len(due)}）")
        for days, topic, _, history in due:
            print(f"  {topic:<{width}}{overdue_text(days)}，已写 {len(history)} 次，上次评分 {history[-1]['rating']}")
    if later:
        print(f"未到期（{len(later)}）")
        for days, topic, card, _ in later:
            print(f"  {topic:<{width}}{card.due.astimezone():%m-%d}（{days} 天后）")
    if fresh:
        print(f"未开始（{len(fresh)}）")
        print("  " + "  ".join(fresh))
    if not due:
        print("今天没有到期的题目。")
    print("\n开始：uv run drill start [题目]")


# ---------- 开始 ----------

def pick_next_topic() -> str:
    """未指定题目时选逾期最久的。还没写过的题目不自动挑选，由用户自己决定何时开始。"""
    due, fresh, _ = due_rows()
    if due:
        return due[0][1]
    hint = f"还没写过的题目：{'、'.join(fresh)}。" if fresh else ""
    die(f"今天没有到期的题目，请指定题目名。{hint}")


def cmd_start(args: argparse.Namespace) -> None:
    topic = check_topic(args.topic) if args.topic else pick_next_topic()
    solution = solution_path(topic)
    stub = TOPICS / topic / "stub.py"

    # 重置会覆盖实现文件。已提交的旧版本在 git 历史里是安全的，未提交的改动则会丢失。
    if solution.exists() and solution.read_bytes() != stub.read_bytes():
        dirty = git("status", "--porcelain", "--", str(solution)).stdout.strip()
        if dirty and not args.force:
            die(
                f"{solution.relative_to(ROOT)} 有未提交的改动，重置会丢失这些内容。\n"
                f"如果是上次没写完想继续，直接接着写，写完运行 drill done；\n"
                f"如果确定要丢弃重来，加 --force。"
            )

    shutil.copyfile(stub, solution)
    save_session({"topic": topic, "started_at": now().isoformat(), "first_run_passed": None})
    readme = TOPICS / topic / "README.md"
    print(f"已重置 {solution.relative_to(ROOT)}，开始计时。")
    print(f"  要求：{readme.relative_to(ROOT)}")
    print(f"  实现：{solution.relative_to(ROOT)}")
    print("写完后运行：uv run drill done")
    vscode(str(readme), str(solution))


# ---------- 测试与完成 ----------

def run_tests(topic: str) -> bool:
    """跑该题目的测试，并记下本次手写的第一次测试是否通过。"""
    result = subprocess.run([sys.executable, "-m", "pytest", str(TOPICS / topic)], cwd=ROOT)
    if result.returncode == 5:
        die(f"topics/{topic} 下没有收集到任何测试。")
    passed = result.returncode == 0
    session = load_session()
    if session and session["topic"] == topic and session["first_run_passed"] is None:
        session["first_run_passed"] = passed
        save_session(session)
    return passed


def cmd_check(args: argparse.Namespace) -> None:
    topic = resolve_topic(args.topic)
    sys.exit(0 if run_tests(topic) else 1)


def ask_peeked() -> bool:
    if not sys.stdin.isatty():
        die("无法交互提问，请加 --peeked 或 --no-peeked 说明这次是否看过参考。")
    answer = input("这次是否看过参考（旧代码、参考实现、资料、AI 提示）？[y/N] ").strip().lower()
    return answer in {"y", "yes", "是"}


def cmd_done(args: argparse.Namespace) -> None:
    topic = resolve_topic(args.topic)
    session = load_session()
    if session is None or session["topic"] != topic:
        die(f"{topic} 没有正在进行的手写。请先运行 drill start {topic}，从空白骨架重新写。")
    solution = solution_path(topic)
    if solution.read_bytes() == (TOPICS / topic / "stub.py").read_bytes():
        die(f"{solution.relative_to(ROOT)} 还是空白骨架，尚未动笔。")

    passed = run_tests(topic)
    session = load_session()
    if not passed and not args.giveup:
        die(
            "\n测试未通过。修改后再次运行 drill done。\n"
            "如果确实想不起来，运行 drill done --giveup 如实记录，之后会打开参考实现供对照。"
        )

    # 评分由客观事实推出，不需要自己斟酌：
    #   看过参考或放弃 → again；独立写出但第一次测试没过 → hard；独立写出且一次通过 → good
    if args.rating:
        rating = args.rating
    elif not passed:
        rating = "again"
    else:
        peeked = args.peeked if args.peeked is not None else ask_peeked()
        rating = "again" if peeked else ("good" if session["first_run_passed"] else "hard")

    finished = now()
    minutes = round((finished - datetime.fromisoformat(session["started_at"])).total_seconds() / 60)
    record = {
        "topic": topic,
        "time": finished.isoformat(timespec="seconds"),
        "rating": rating,
        "passed": passed,
        "first_run_passed": bool(session["first_run_passed"]),
        "minutes": minutes,
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    SESSION.unlink()

    card, history = card_of(topic, load_reviews())
    print(f"\n评分 {rating}，用时 {minutes} 分钟，这是第 {len(history)} 次。")
    print(f"下次复习：{card.due.astimezone():%Y-%m-%d}（{days_until_due(card)} 天后）")

    paths = [str(TOPICS / topic), str(LOG)]
    git("add", "--", *paths)
    commit = git("commit", "-m", f"review({topic}): {rating}，第 {len(history)} 次", "--", *paths)
    if commit.returncode != 0:
        print(f"\n复习已记录，但 git 提交失败，请手动提交：\n{commit.stderr.strip() or commit.stdout.strip()}")
    elif git("rev-parse", "--abbrev-ref", "@{u}").returncode == 0:
        push = git("push")
        print("已提交并推送。" if push.returncode == 0 else f"已提交，但推送失败：\n{push.stderr.strip()}")
    else:
        print("已提交（尚未设置远程仓库，未推送）。")

    # 写完之后才对照：先看与参考实现的差别，再看与自己上一次的差别
    if not args.no_open:
        diff_with_reference(topic)
        if len(history) > 1:
            diff_with_previous(topic)
    print(f"再次查看对比：uv run drill diff {topic} --ref（与参考实现）"
          + (f"，uv run drill diff {topic}（与上一次）" if len(history) > 1 else ""))


# ---------- 对比 ----------

def show_diff(left: Path, right: Path) -> None:
    """优先在 VSCode 的对比视图中打开，否则退回终端输出。"""
    if not vscode("--diff", str(left), str(right)):
        subprocess.run(["git", "diff", "--no-index", "--", str(left), str(right)], cwd=ROOT)


def diff_with_reference(topic: str) -> None:
    reference = reference_path(topic)
    if not reference.exists():
        print(f"{topic} 没有参考实现。")
        return
    show_diff(reference, solution_path(topic))


def diff_with_previous(topic: str) -> None:
    solution = solution_path(topic)
    commits = git("log", "--format=%H", "--", str(solution)).stdout.split()
    if len(commits) < 2:
        print(f"{topic} 目前只有 {len(commits)} 次提交记录，还没有可对比的上一版。")
        return
    # 把上一版从 git 历史中取出来放到临时位置，才能交给 VSCode 做对比
    previous = STATE / "previous" / f"{topic}@{commits[1][:7]}.py"
    previous.parent.mkdir(parents=True, exist_ok=True)
    relative = solution.relative_to(ROOT).as_posix()
    previous.write_text(git("show", f"{commits[1]}:{relative}").stdout, encoding="utf-8")
    show_diff(previous, solution)


def cmd_diff(args: argparse.Namespace) -> None:
    topic = resolve_topic(args.topic)
    session = load_session()
    if session and session["topic"] == topic:
        die("这道题正在手写中，现在对比等于看参考。写完运行 drill done 之后再对比。")
    if not solution_path(topic).exists():
        die(f"{topic} 还没有写过。")
    (diff_with_reference if args.ref else diff_with_previous)(topic)


# ---------- 检验测试本身 ----------

def verify_cases(topic: str) -> list[tuple[str, str, bool]]:
    """列出（名称、源码、是否应当通过）：参考实现应通过，mutants.toml 中的每种错误写法应失败。"""
    source = reference_path(topic).read_text(encoding="utf-8")
    cases = [("参考实现", source, True)]
    mutants_file = TOPICS / topic / "mutants.toml"
    if mutants_file.exists():
        for mutant in tomllib.loads(mutants_file.read_text(encoding="utf-8")).get("mutant", []):
            if mutant["old"] not in source:
                die(f"{topic}/mutants.toml：在参考实现中找不到「{mutant['name']}」要替换的片段：{mutant['old']!r}")
            cases.append((mutant["name"], source.replace(mutant["old"], mutant["new"], 1), False))
    return cases


def run_case(topic: str, source: str) -> tuple[bool, str]:
    """在临时目录中把给定源码当作用户的实现，跑该题目的测试。"""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / topic
        shutil.copytree(TOPICS / topic, work, ignore=shutil.ignore_patterns("__pycache__"))
        (work / f"{topic}.py").write_text(source, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(work), "-q", "-p", "no:cacheprovider", "--tb=line"],
            cwd=work, capture_output=True, text=True,
        )
        return result.returncode == 0, (result.stdout + result.stderr).strip()


def cmd_verify(args: argparse.Namespace) -> None:
    topics = [check_topic(t) for t in args.topics] or all_topics()
    missing = [t for t in topics if not reference_path(t).exists()]
    jobs = [(t, name, source, expected)
            for t in topics if t not in missing
            for name, source, expected in verify_cases(t)]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(lambda job: run_case(job[0], job[2]), jobs))

    ok = not missing
    width = max(len(t) for t in topics) + 2
    for topic in topics:
        if topic in missing:
            print(f"{topic:<{width}}缺少参考实现 {topic}_ref.py")
            continue
        rows = [(name, expected, passed, output)
                for (t, name, _, expected), (passed, output) in zip(jobs, results) if t == topic]
        reference_ok = rows[0][2]
        mutants = rows[1:]
        caught = [r for r in mutants if not r[2]]
        print(f"{topic:<{width}}参考实现{'通过' if reference_ok else '未通过'}；"
              f"错误写法被拦住 {len(caught)}/{len(mutants)}")
        if not reference_ok:
            ok = False
            print("    " + rows[0][3].replace("\n", "\n    "))
        for name, _, passed, _ in mutants:
            if passed:
                ok = False
                print(f"    未被拦住：{name}")
    sys.exit(0 if ok else 1)


# ---------- 入口 ----------

def main() -> None:
    parser = argparse.ArgumentParser(prog="drill", description="手写练习的复习调度工具")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("status", help="查看哪些题目到期（默认）")
    p.add_argument("--due-only", action="store_true", help="只输出到期的题目，供定时提醒使用")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("start", help="开始一次手写：把实现文件重置为空白骨架")
    p.add_argument("topic", nargs="?", help="省略则自动选逾期最久的题目")
    p.add_argument("--force", action="store_true", help="丢弃实现文件中未提交的改动")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("check", help="只跑测试，不记录不提交")
    p.add_argument("topic", nargs="?")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("done", help="写完：跑测试、评分、记录、提交，然后打开对比")
    p.add_argument("topic", nargs="?")
    p.add_argument("--peeked", action=argparse.BooleanOptionalAction, default=None,
                   help="这次是否看过参考；省略则交互提问")
    p.add_argument("--giveup", action="store_true", help="测试没过也如实记录为 again")
    p.add_argument("--rating", choices=list(RATINGS), help="手动指定评分，覆盖自动判定")
    p.add_argument("--no-open", action="store_true", help="完成后不自动打开对比")
    p.set_defaults(func=cmd_done)

    p = sub.add_parser("diff", help="与上一次手写的版本对比")
    p.add_argument("topic", nargs="?")
    p.add_argument("--ref", action="store_true", help="改为与参考实现对比")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("verify", help="检验测试本身：参考实现应通过，各种错误写法应被拦住")
    p.add_argument("topics", nargs="*", help="省略则检验全部题目")
    p.set_defaults(func=cmd_verify)

    # 输出被重定向时也按行刷新，保证自身的输出与 pytest、git 等子进程的输出顺序一致
    sys.stdout.reconfigure(line_buffering=True)
    args = parser.parse_args()
    (args.func if args.command else cmd_status)(args)
