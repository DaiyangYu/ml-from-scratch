"""drill：手写练习的复习调度工具。

    uv run drill                查看哪些题目到期
    uv run drill start [题目]   开始一次手写（把实现文件重置为空白骨架，并在 VSCode 中打开）
    uv run drill done  [题目]   写完：跑测试、评分、记录、提交，然后打开对比
    uv run drill check [题目]   只跑测试，不记录不提交
    uv run drill diff  [题目]   与上一次手写的版本对比；加 --ref 与参考实现对比
    uv run drill verify [题目]  检验测试本身：参考实现应通过，各种错误写法应被拦住
    uv run drill pause  题目…   暂停复习计划：不再出现在到期列表和每日提醒中；--all 暂停全部
    uv run drill resume 题目…   恢复复习计划；--all 恢复全部

调度状态不单独存储：reviews.jsonl 是只追加的记录（复习、暂停、恢复），
每次都由它重新推算出各题目的到期时间和是否暂停。

日期一律按“学习日”计算：北京时间，以凌晨 4 点为一天的开始。间隔推算、到期显示和
每日提醒用的是同一套日期，所以显示“今天到期”的题目，当天任何时候写都按隔了整天处理。
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
from datetime import date, datetime, timedelta, timezone
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

# 学习日的时区和起点。时区写死为北京时间（没有夏令时，用固定偏移即可），
# 这样由记录推算的结果不随运行所在机器的时区变化；以凌晨 4 点为界，过了零点才写完的仍算前一天。
STUDY_TZ = timezone(timedelta(hours=8))
DAY_STARTS_AT_HOUR = 4


# ---------- 基础工具 ----------

def now() -> datetime:
    override = os.environ.get("DRILL_NOW")  # 仅用于测试调度逻辑
    if override:
        return datetime.fromisoformat(override).astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def study_day(moment: datetime) -> date:
    """某个时刻属于哪个学习日。"""
    return (moment.astimezone(STUDY_TZ) - timedelta(hours=DAY_STARTS_AT_HOUR)).date()


def replay_time(day: date) -> datetime:
    """交给 FSRS 的复习时间：学习日当天的固定时刻。

    FSRS 把两次复习的时间差向下取整为天数，不满 24 小时算作同一天内的重复练习，稳定性几乎不增长。
    如果直接传真实时间戳，前一天晚上写、第二天白天复习就会落入这种情况。
    折算之后，它算出的天数恰好等于两个学习日之差。
    """
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


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


def load_log() -> list[dict]:
    """读出全部记录。带 rating 的是一次复习，带 event 的是暂停（pause）或恢复（resume）。"""
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_log(*records: dict) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def card_of(topic: str, log: list[dict]) -> tuple[Card | None, list[dict]]:
    """由复习记录推算某题目当前的调度状态。

    暂停与恢复不参与推算：暂停只是不提醒，遗忘并不会停止，所以到期时间不顺延。
    恢复后隔了多久才复习，由下一次复习记录的时间如实交给 FSRS 处理。

    首次学习也不参与推算，只把到期日定在下一个学习日（见 is_first_learning）。
    """
    history = [r for r in log if r["topic"] == topic and "rating" in r]
    if not history:
        return None, history
    days = [study_day(datetime.fromisoformat(r["time"])) for r in history]
    skip = 1 if is_first_learning(history[:1]) else 0
    card = Card(due=replay_time(days[0] + timedelta(days=1))) if skip else Card()
    for r, day in zip(history[skip:], days[skip:]):
        card, _ = SCHEDULER.review_card(card, RATINGS[r["rating"]], replay_time(day))
    return card, history


def is_first_learning(history: list[dict]) -> bool:
    """这道题是否只有一条评分为 again 的记录，即刚完成首次学习。

    第一次写一道题时看参考是在学习，不是在回忆。FSRS 用第一次评分给题目定难度，而且此后
    连续 good 也几乎不会让它下降；如果把首次学习当作一次 again，所有题目都会从同一个偏高的
    难度起步，难度不再反映题目本身。所以第一条 again 不交给 FSRS，
    从下一次由空白独立重写开始评分。
    """
    return len(history) == 1 and history[0]["rating"] == "again"


def paused_since(topic: str, log: list[dict]) -> datetime | None:
    """题目处于暂停状态时返回暂停的时间，否则返回 None。"""
    since = None
    for r in log:
        if r["topic"] == topic and "event" in r:
            since = datetime.fromisoformat(r["time"]) if r["event"] == "pause" else None
    return since


def commit_and_push(paths: list[Path], message: str) -> None:
    """提交指定路径并推送。失败时只提示不中断，因为记录已经写入文件。"""
    names = [str(p) for p in paths]
    git("add", "--", *names)
    commit = git("commit", "-m", message, "--", *names)
    if commit.returncode != 0:
        print(f"\n已记录，但 git 提交失败，请手动提交：\n{commit.stderr.strip() or commit.stdout.strip()}")
    elif git("rev-parse", "--abbrev-ref", "@{u}").returncode == 0:
        push = git("push")
        print("已提交并推送。" if push.returncode == 0 else f"已提交，但推送失败：\n{push.stderr.strip()}")
    else:
        print("已提交（尚未设置远程仓库，未推送）。")


def due_day(card: Card) -> date:
    """到期的学习日。card.due 由 replay_time 加整数天得到，日期部分就是学习日。"""
    return card.due.date()


def days_until_due(card: Card) -> int:
    """距离到期还有几个学习日，负数表示已逾期。"""
    return (due_day(card) - study_day(now())).days


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

def due_rows() -> tuple[list, list, list, list]:
    """返回（到期、未开始、未到期、已暂停）四组题目。"""
    log = load_log()
    due, fresh, later, paused = [], [], [], []
    for topic in all_topics():
        card, history = card_of(topic, log)
        if card is None:
            fresh.append(topic)
            continue
        since = paused_since(topic, log)
        if since is not None:
            paused.append((topic, since, card))
            continue
        days = days_until_due(card)
        (due if days <= 0 else later).append((days, topic, card, history))
    return sorted(due), fresh, sorted(later), paused


def overdue_text(days: int) -> str:
    return "今天到期" if days == 0 else f"已逾期 {-days} 天"


def schedule_text(card: Card) -> str:
    days = days_until_due(card)
    return overdue_text(days) if days <= 0 else f"{due_day(card):%m-%d} 到期（{days} 天后）"


def cmd_status(args: argparse.Namespace) -> None:
    topics = all_topics()
    due, fresh, later, paused = due_rows()
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
            print(f"  {topic:<{width}}{due_day(card):%m-%d}（{days} 天后）")
    if paused:
        print(f"已暂停（{len(paused)}）")
        for topic, since, card in paused:
            print(f"  {topic:<{width}}{study_day(since):%m-%d} 起暂停，原计划：{schedule_text(card)}")
    if fresh:
        print(f"未开始（{len(fresh)}）")
        print("  " + "  ".join(fresh))
    if not due:
        print("今天没有到期的题目。")
    print("\n开始：uv run drill start [题目]")


# ---------- 开始 ----------

def pick_next_topic() -> str:
    """未指定题目时选逾期最久的。还没写过的和已暂停的题目不自动挑选，由用户自己决定何时开始。"""
    due, fresh, _, _ = due_rows()
    if due:
        return due[0][1]
    hint = f"还没写过的题目：{'、'.join(fresh)}。" if fresh else ""
    die(f"今天没有到期的题目，请指定题目名。{hint}")


def cmd_start(args: argparse.Namespace) -> None:
    topic = check_topic(args.topic) if args.topic else pick_next_topic()
    solution = solution_path(topic)
    stub = TOPICS / topic / "stub.py"

    # 同一时间只记录一次手写。开始另一道题会覆盖进行中那一次的开始时间和测试记录。
    session = load_session()
    if session and session["topic"] != topic and not args.force:
        die(
            f"{session['topic']} 的手写还在进行中，开始另一道题会丢掉它的开始时间和测试记录。\n"
            f"先把它写完，运行 drill done（写不出来就运行 drill done --giveup）；\n"
            f"如果确定不要那一次的记录，加 --force。已经写下的代码不受影响。"
        )

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
    save_session({"topic": topic, "started_at": now().isoformat(), "first_run_passed": None, "failed_runs": 0})
    readme = TOPICS / topic / "README.md"
    print(f"已重置 {solution.relative_to(ROOT)}，开始计时。")
    print(f"  要求：{readme.relative_to(ROOT)}")
    print(f"  实现：{solution.relative_to(ROOT)}")
    print("写完后运行：uv run drill done")
    vscode(str(readme), str(solution))


# ---------- 测试与完成 ----------

def run_tests(topic: str) -> bool:
    """跑该题目的测试，并记下本次手写的第一次测试是否通过、一共失败过几次。"""
    result = subprocess.run([sys.executable, "-m", "pytest", str(TOPICS / topic)], cwd=ROOT)
    if result.returncode == 5:
        die(f"topics/{topic} 下没有收集到任何测试。")
    passed = result.returncode == 0
    session = load_session()
    if session and session["topic"] == topic:
        if session["first_run_passed"] is None:
            session["first_run_passed"] = passed
        if not passed:
            session["failed_runs"] = session.get("failed_runs", 0) + 1
        save_session(session)
    return passed


def cmd_check(args: argparse.Namespace) -> None:
    topic = resolve_topic(args.topic)
    sys.exit(0 if run_tests(topic) else 1)


def ask(question: str, flag: str, meaning: str) -> bool:
    if not sys.stdin.isatty():
        die(f"无法交互提问，请加 --{flag} 或 --no-{flag} 说明{meaning}。")
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes", "是"}


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

    # 评分由事实推出：
    #   看过参考或放弃 → again；
    #   独立写出，但出过概念性错误（算法、公式、定义写错，由测试或写完后的对照发现）→ hard；
    #   独立写出，没有出错，或只有笔误、接口细节之类的小错 → good。
    # 测试失败本身不降级，否则会让人不敢随手跑测试。easy 只能用 --rating 手动指定。
    failed_runs = session.get("failed_runs", 0)
    peeked, conceptual = args.peeked, args.conceptual
    if args.rating:
        rating = args.rating
    elif not passed:
        rating = "again"
    else:
        if peeked is None:
            peeked = ask("这次是否看过参考（旧代码、参考实现、资料、AI 提示）？", "peeked", "这次是否看过参考")
        if peeked:
            rating = "again"
        else:
            if conceptual is None and failed_runs > 0:
                conceptual = ask(
                    f"测试失败过 {failed_runs} 次。其中有没有概念性错误（算法、公式、定义写错，而不是笔误或接口细节）？",
                    "conceptual", "这次是否出过概念性错误")
            # 测试没有失败过、也没有用参数指明时，按没有概念性错误评分，但 conceptual 保持为 None，
            # 不写进记录：失败次数只统计经过 drill 跑的测试，没有问过就不能当作确认过的事实。
            rating = "hard" if conceptual else "good"

    finished = now()
    minutes = round((finished - datetime.fromisoformat(session["started_at"])).total_seconds() / 60)
    record = {
        "topic": topic,
        "time": finished.isoformat(timespec="seconds"),
        "rating": rating,
        "passed": passed,  # 为 False 表示放弃
        "first_run_passed": bool(session["first_run_passed"]),
        "failed_runs": failed_runs,
        "minutes": minutes,
    }
    # 评分所依据的事实也一并记下，以后调整评分规则时可以按新规则重新判定。没有问到的不记。
    if peeked is not None:
        record["peeked"] = peeked
    if conceptual is not None:
        record["conceptual"] = conceptual
    append_log(record)
    SESSION.unlink()

    log = load_log()
    card, history = card_of(topic, log)
    if is_first_learning(history):
        print(f"\n首次学习，用时 {minutes} 分钟。这一次不参与间隔推算，下一次从空白独立重写时才开始评分。")
    else:
        print(f"\n评分 {rating}，用时 {minutes} 分钟，这是第 {len(history)} 次。")
    print(f"下次复习：{due_day(card):%Y-%m-%d}（{days_until_due(card)} 天后）")
    if paused_since(topic, log) is not None:
        print(f"这道题仍处于暂停状态，到期时不会提醒。恢复：uv run drill resume {topic}")

    commit_and_push([TOPICS / topic, LOG], f"review({topic}): {rating}，第 {len(history)} 次")

    # 写完之后才对照：先看与参考实现的差别，再看与自己上一次的差别
    if not args.no_open:
        diff_with_reference(topic)
        if len(history) > 1:
            diff_with_previous(topic)
    print(f"再次查看对比：uv run drill diff {topic} --ref（与参考实现）"
          + (f"，uv run drill diff {topic}（与上一次）" if len(history) > 1 else ""))


# ---------- 暂停与恢复 ----------

def set_paused(args: argparse.Namespace, pause: bool) -> None:
    """追加暂停或恢复事件。提交并推送之后，每日提醒才会随之变化。"""
    verb = "暂停" if pause else "恢复"
    log = load_log()
    written = [t for t in all_topics() if card_of(t, log)[0] is not None]
    paused = [t for t in written if paused_since(t, log) is not None]
    candidates = [t for t in written if t not in paused] if pause else paused

    if args.all and args.topics:
        die("--all 不能与题目名同时使用。")
    if args.all:
        topics = candidates
        if not topics:
            die("没有正在复习的题目可以暂停。" if pause else "没有处于暂停状态的题目。")
    elif args.topics:
        topics = list(dict.fromkeys(check_topic(t) for t in args.topics))
    else:
        hint = f"可以{verb}的题目：{'、'.join(candidates)}。" if candidates else f"目前没有可以{verb}的题目。"
        die(f"请指定题目名，或加 --all。{hint}")

    changed = []
    for topic in topics:
        if topic not in written:
            print(f"{topic}：还没写过，没有复习计划，不需要{verb}。")
        elif pause and topic in paused:
            print(f"{topic}：{study_day(paused_since(topic, log)):%Y-%m-%d} 起已经暂停。")
        elif not pause and topic not in paused:
            print(f"{topic}：没有暂停。")
        else:
            changed.append(topic)
    if not changed:
        return

    time = now().isoformat(timespec="seconds")
    append_log(*({"topic": t, "time": time, "event": "pause" if pause else "resume"} for t in changed))
    for topic in changed:
        # 到期时间在暂停期间不顺延，恢复后按原计划继续，已过期的立即进入到期列表
        print(f"已{verb} {topic}，原计划：{schedule_text(card_of(topic, log)[0])}")
    if pause:
        print("暂停期间不出现在到期列表和每日提醒中，仍可以用 drill start <题目> 主动手写。")
        print("恢复：uv run drill resume <题目>")

    summary = "、".join(changed) if len(changed) <= 3 else f"{changed[0]} 等 {len(changed)} 道题"
    commit_and_push([LOG], f"{'pause' if pause else 'resume'}: {summary}")


def cmd_pause(args: argparse.Namespace) -> None:
    set_paused(args, pause=True)


def cmd_resume(args: argparse.Namespace) -> None:
    set_paused(args, pause=False)


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
    p.add_argument("--force", action="store_true",
                   help="丢弃实现文件中未提交的改动；另一道题的手写还在进行时，放弃那一次的记录")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("check", help="只跑测试，不记录不提交")
    p.add_argument("topic", nargs="?")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("done", help="写完：跑测试、评分、记录、提交，然后打开对比")
    p.add_argument("topic", nargs="?")
    p.add_argument("--peeked", action=argparse.BooleanOptionalAction, default=None,
                   help="这次是否看过参考；省略则交互提问")
    p.add_argument("--conceptual", action=argparse.BooleanOptionalAction, default=None,
                   help="这次是否出过概念性错误（算法、公式、定义写错）；省略时，测试失败过才交互提问")
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

    p = sub.add_parser("pause", help="暂停复习计划：不再出现在到期列表和每日提醒中")
    p.add_argument("topics", nargs="*")
    p.add_argument("--all", action="store_true", help="暂停所有已经写过的题目")
    p.set_defaults(func=cmd_pause)

    p = sub.add_parser("resume", help="恢复复习计划：按原计划的到期时间继续，不顺延")
    p.add_argument("topics", nargs="*")
    p.add_argument("--all", action="store_true", help="恢复所有已暂停的题目")
    p.set_defaults(func=cmd_resume)

    # 输出被重定向时也按行刷新，保证自身的输出与 pytest、git 等子进程的输出顺序一致
    sys.stdout.reconfigure(line_buffering=True)
    args = parser.parse_args()
    (args.func if args.command else cmd_status)(args)
