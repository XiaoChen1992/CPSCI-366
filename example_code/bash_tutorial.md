# 1) Shell & Filesystem Essentials
## Navigation & inspection
```bash
pwd                         # print working directory
ls -lah                     # list (long, all, human sizes)
cd /path/to/dir             # change directory
tree -L 2                   # directory tree (install: apt/yum/brew install tree)
```
## Create, move, copy, delete
```bash
mkdir -p data/raw           # -p makes parent dirs as needed
cp src.py backup/src.py     # copy file
mv file.txt dir/            # move or rename
rm file.txt                 # delete file
rm -rf old_dir              # recursive delete (⚠ irreversible)
```

## View/edit files
```bash
cat file.log                # print whole file
head -n 50 file.log         # first 50 lines
tail -n 50 file.log         # last 50 lines
less file.log               # pager; q to quit
nano file.txt               # simple terminal editor
```

## Disk usage
```bash
df -h                       # disk free by mount
du -sh *                    # size of items in current dir
```

## Search (text & files)
```bash
grep -n "pattern" file.log                # find text (show line numbers)
grep -nri "todo" src/                     # recursive, case-insensitive
find . -name "*.py"                       # find by name
find . -type f -mtime -1                  # files modified <1 day
```

## Permissions (quick refs)
```bash
chmod +x script.sh          # make executable
chmod 640 file              # rw-r----- (u=rw,g=r,o=-)
# Google other permission settings
```

# 2) Processes: list, filter, and kill
```bash
ps aux | less                               # all processes
ps -eo pid,user,pcpu,pmem,etime,cmd | head  # custom columns
pgrep -fa python                            # find PIDs by name (+ cmdline)
kill <PID>                                  # gentle stop (SIGTERM)
kill -9 <PID>                               # force kill (SIGKILL) if needed
pkill -f "train.py --exp exp1"              # kill by matching full cmdline
```

**Live monitors:**
```bash
top                # CPU/mem (built-in)
htop               # friendlier top (install first)
free -h            # memory snapshot
watch -n 1 'nvidia-smi'   # rerun nvidia-smi every 1s
```

# 3) GPU Monitoring (NVIDIA)
**Quick status**
```bash
nvidia-smi                       # GPUs, memory, running processes
nvidia-smi -L                    # list GPUs
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
watch -n 2 nvidia-smi            # refresh every 2s
```

# 4) Run Long Jobs in Background
**Keep running after logout → `nohup`**
```bash
# Basic
nohup python train.py > train.log 2>&1 &
#   - nohup: ignore hangup (keeps running after logout)
#   - > train.log: write stdout to log
#   - 2>&1: send stderr to same log
#   - &: run in background

# If you forget redirection, output goes to ./nohup.out by default.
```

# 5) Logging: save, rotate, and inspect
**Redirects & patterns**
```bash
cmd > out.log               # stdout to file (overwrite)
cmd >> out.log              # append
cmd > out.log 2>&1          # stdout+stderr into file
cmd &> out.log              # bash-only shorthand for both
```


**Stream to screen and save:**
```bash
# Good when running foreground:
python -u train.py 2>&1 | tee -a train.log

# With nohup:
nohup bash -lc 'python -u train.py 2>&1 | tee -a train.log' &
```

**Check progress**
```bash
tail -n 100 train.log           # last 100 lines
tail -f train.log               # follow (live)
less +F train.log               # follow inside less (Ctrl-C to stop follow)
grep -n "loss" train.log        # find lines with 'loss' (+ line nums)
grep -nE "loss|acc" train.log   # regex OR
```

# 6) Quick Networking & Transfer
```bash
ssh user@host                      # remote login
scp file.txt user@host:~/          # copy to remote
rsync -avh --progress src/ dest/   # robust sync (retries, deltas)
```

# 7) Helpful One-Liners & Tips
```bash
which python            # path to executable
type python             # how the shell resolves a command/alias
env | sort              # environment variables
alias ll='ls -lah'      # quick alias
history | grep git      # search your command history
watch -n 5 "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv"
```