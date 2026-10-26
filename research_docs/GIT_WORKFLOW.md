# 公开仓库与日常 Git 工作流

## Remote 与分支

- `origin` 是研究 fork：`https://github.com/mao455/bcrl_visualTactile.git`。
- `upstream` 是原项目：`https://github.com/kushal2000/play2perfect.git`。
- `main` 跟踪 `origin/main`；同步原项目时显式使用 `upstream`，不要把 upstream 当作推送目标。
- 当前本地 `main` 从原仓库 `ff2fc62` 开始。首次公开提交应保留这段历史，再增加研究扩展。

检查 remote 和分支：

```bash
git remote -v
git branch -vv
git fetch --prune upstream origin
```

## 每次研究修改

1. 开始前阅读 `AGENTS.md`、`research_docs/README.md` 和相关合约；确认没有未解释的本地修改。
2. 从 `main` 建立功能分支，例如 `feature/correction-dataset`。一次分支只做一个可说明的研究功能。
3. 修改代码、测试和对应 Markdown 记录。记录输入来源、维度、前后行为和真实测试结果；不要把 checkpoint、轨迹或日志当作代码提交。
4. 使用项目环境运行必要的测试：

   ```bash
   bash visuotactile/run_cpu.sh -m pytest -q tests/test_visuotactile.py
   git diff --check
   ```

5. 提交前只显式加入审计过的路径：

   ```bash
   git status --short
   git add README.md AGENTS.md research_docs visuotactile tests pyproject.toml setup.py .gitignore
   git diff --cached --stat
   git diff --cached --check
   git status --short
   git commit -m "Add <one focused research change>"
   ```

   不使用未经检查的 `git add -A`。确认 staged 列表中没有 `outputs/`、`pretrained_*`、`.venv*`、私有配置、数据、模型、视频、密钥或其他工作目录文件。

6. 默认只做本地 commit。只有用户明确授权本次推送后，才执行：

   ```bash
   git push -u origin <feature-branch>
   ```

   永远不要执行 `git push upstream`。推送后报告 commit ID、remote、分支、结果和 GitHub URL。

## 同步原项目

先获取并查看差异：

```bash
git fetch upstream
git log --oneline --left-right main...upstream/main
git diff --stat main...upstream/main
```

在功能分支上合并或 rebase upstream，并重新运行测试和文档审计。涉及原 task、资产、控制、随机化、reset 或 RL 的冲突必须先解释其物理/信息边界，再提交。不要用强制 push 改写公开的 `main`。

## 公开内容检查

`.gitignore` 忽略实验输出、训练 checkpoint、轨迹数据、缓存、虚拟环境、常见凭据和本地日志。忽略规则不会从历史中删除已经提交的文件；若发现误提交的 secret，先停止推送并按安全流程撤销/轮换。公开前运行：

```bash
git status --short --ignored
git ls-files | rg '(^|/)(\.env|.*\.(pem|key|p12|pfx|pt|pth|ckpt|safetensors|onnx|engine|h5|hdf5|npy|npz|log))$'
rg -n -i --hidden --glob '!.git/**' --glob '!outputs/**' --glob '!pretrained_*' --glob '!.venv*/**' \
  -e 'api[_-]?key|access[_-]?token|secret[_-]?key|password\\s*[:=]|BEGIN (RSA|OPENSSH|EC|PRIVATE) KEY|ghp_|github_pat_|sk-' .
```

仓库保留根目录原 Play2Perfect MIT `LICENSE`。新增 vendored Diffusion Policy 文件保留 `visuotactile/models/vendor/LICENSE`，来源、commit 和 hash 记录在 `PROVENANCE.md`。论文、代码和数据的许可不因“参考过”而自动获得；新增第三方文件必须先核对许可并保留声明。

## 当前首次公开发布状态

截至 2026-10-27，首次公开同步已完成：commit `159599f0575d8f61fbf21b615e2b02d0fd4168ee` 已推送到 `origin/main`。GitHub 地址为 https://github.com/mao455/bcrl_visualTactile 。首次推送没有包含 `outputs/`、预训练模型、实验轨迹、虚拟环境或私有配置。
