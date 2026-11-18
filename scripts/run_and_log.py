"""Run a command, show its output live in the terminal AND save it to a log file.
Usage: python scripts/run_and_log.py <logfile> <command ...>
Example: python scripts/run_and_log.py results/bayes_opt_log.txt python -u scripts/bayes_opt.py --trials 40
"""
import subprocess, sys

log, cmd = sys.argv[1], sys.argv[2:]
with open(log, "w", encoding="utf-8") as f:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        f.write(line)
        f.flush()
    sys.exit(p.wait())
