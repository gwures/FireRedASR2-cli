# FireRedASR 命令行使用手册

## 目录
- [什么是命令行](#什么是命令行)
- [准备工作](#准备工作)
- [开始使用 cli.py](#开始使用-clipy)
- [深入了解 cli.py 的所有选项](#深入了解-clipy-的所有选项)
- [开始使用 test_direct.py](#开始使用-test_directpy)
- [常见问题解答](#常见问题解答)
- [完整示例参考](#完整示例参考)

---

## 什么是命令行

### 🎯 简单理解

命令行（也叫终端、CMD、PowerShell）就是一个用文字指令操作电脑的工具。就像给电脑发微信，你打字告诉它做什么，它打字回复你结果。

### 🖥️ 怎么打开命令行（Windows）

**方法一：使用 PowerShell（推荐）**
1. 在文件夹空白处，按住 `Shift` 键，右键点击
2. 选择「在此处打开 PowerShell 窗口」或「在终端中打开」

**方法二：使用搜索**
1. 按下 `Win` 键（键盘上的 Windows 徽标键）
2. 输入 `PowerShell` 或 `cmd`
3. 按回车键打开

### 📝 最基本的命令（必须学会）

| 命令 | 说明 | 例子 |
|------|------|------|
| `cd 文件夹路径` | 进入某个文件夹 | `cd d:\fireredasr` |
| `dir` 或 `ls` | 查看当前文件夹里的文件 | `dir` |
| `python 文件名.py` | 运行 Python 脚本 | `python cli.py --help` |

**提示：** 按键盘上的 `↑` 和 `↓` 可以快速切换之前用过的命令！

---

## 准备工作

### ✅ 第一步：检查 Python 是否已安装

在命令行输入：
```
python --version
```

如果看到类似 `Python 3.10.x` 这样的输出，说明已安装。如果没有，请先安装 Python 3.8 或更高版本。

### ✅ 第二步：进入项目文件夹

假设你的项目在 `d:\fireredasr`，在命令行输入：
```
cd d:\fireredasr
```

### ✅ 第三步：查看帮助信息

任何时候不确定怎么用，都可以用 `--help` 查看帮助：

```
python cli.py --help
```

```
python test_direct.py --help
```

---

## 开始使用 cli.py

### 🤔 cli.py 是做什么的？

`cli.py` 是 FireRedASR 的主要命令行工具，用来把音频/视频文件转换成文字（语音识别）。

### 🚀 最最简单的例子（必看！）

假设你有一个音频文件叫 `my_audio.mp3`，放在 `d:\fireredasr` 文件夹里。

**方法：自动优化批次大小（推荐新手用）**
```
python cli.py -f my_audio.mp3 --ab 16
```

就这么简单！运行后等待一会儿，识别结果会自动保存在 `results` 文件夹里。

---

## cli.py 详细使用指南

### 📂 输入文件：指定要识别的文件

#### 方式一：处理单个或多个文件（用 `-f`）

```bash
# 处理 1 个文件
python cli.py -f 录音.mp3 --ab 16

# 处理多个文件（空格分开）
python cli.py -f 录音1.mp3 录音2.wav 采访.m4a --ab 16
```

#### 方式二：处理整个文件夹（用 `-d`）

```bash
# 处理当前文件夹里的所有音频文件
python cli.py -d . --ab 16

# 处理指定文件夹
python cli.py -d d:\我的录音 --ab 16

# 递归处理：连同子文件夹里的文件一起处理（加 -r）
python cli.py -d d:\我的录音 -r --ab 16
```

### 🎛️ 批次大小设置（核心参数）

#### 什么是批次大小？
简单说，批次大小就是一次处理多少段音频。越大越快，但越占显存。

#### 两个选择：

**选项 A：自动优化（推荐！用 `--ab`）**
```bash
# 从 16 开始自动调整
python cli.py -f 录音.mp3 --ab 16

# 从 32 开始自动调整（更快，但需要显存够）
python cli.py -f 录音.mp3 --ab 32
```

**选项 B：固定大小（用 `--bs`）**
```bash
# 固定用 8
python cli.py -f 录音.mp3 --bs 8

# 固定用 16
python cli.py -f 录音.mp3 --bs 16
```

**新手建议：** 先用 `--ab 16`，如果报错显存不足，就改成 `--ab 8`。

### ✨ 高级功能选项

#### 1. 标点预测（`--pu`）
自动给识别结果加标点符号（逗号、句号等）

```bash
python cli.py -f 录音.mp3 --ab 16 --pu
```

#### 2. 时间戳（`--ts`）
输出每句话的开始和结束时间

```bash
python cli.py -f 录音.mp3 --ab 16 --ts
```

#### 3. 禁用 FP16（`--nfp`）
如果显卡不支持 FP16，加上这个参数

```bash
python cli.py -f 录音.mp3 --ab 16 --nfp
```

### 📁 输出设置

#### 自定义输出文件夹（`-o`）
默认保存在 `results` 文件夹，你可以改成别的：

```bash
python cli.py -f 录音.mp3 --ab 16 -o 我的输出文件夹
```

### 🔇 日志输出控制

| 参数 | 效果 |
|------|------|
| `-v` 或 `--verbose` | 显示详细的调试信息 |
| `-q` 或 `--quiet` | 只显示进度和错误，安静模式 |

```bash
# 安静模式，不显示太多日志
python cli.py -f 录音.mp3 --ab 16 -q
```

---

## cli.py 所有参数速查表

| 参数 | 简写 | 说明 | 例子 |
|------|------|------|------|
| `--files` | `-f` | 输入文件 | `-f a.mp3 b.wav` |
| `--directory` | `-d` | 输入目录 | `-d ./audio` |
| `--recursive` | `-r` | 递归搜索子目录 | `-d ./audio -r` |
| `--output-dir` | `-o` | 输出目录 | `-o my_output` |
| `--auto-batch` | `--ab` | 自动批次起始值 | `--ab 16` |
| `--batch-size` | `--bs` | 固定批次大小 | `--bs 8` |
| `--auto-batch-range` | `--abr` | 显存目标区间 | `--abr 0.75-0.95` |
| `--punc` | `--pu` | 启用标点预测 | `--pu` |
| `--no-punc` | `--npu` | 禁用标点预测（默认） | `--npu` |
| `--timestamp` | `--ts` | 输出时间戳 | `--ts` |
| `--no-timestamp` | `--nts` | 不输出时间戳（默认） | `--nts` |
| `--fp16` | `--fp` | 启用 FP16（默认） | `--fp` |
| `--no-fp16` | `--nfp` | 禁用 FP16 | `--nfp` |
| `--verbose` | `-v` | 详细日志 | `-v` |
| `--quiet` | `-q` | 安静模式 | `-q` |

---

## 开始使用 test_direct.py

### 🤔 test_direct.py 是做什么的？

`test_direct.py` 是性能测试工具，用来测试不同批次大小下的识别速度，帮你找到最快的设置。

### 🚀 最简单的测试例子

```bash
# 测试固定批次大小 1 到 10
python test_direct.py -f 录音.mp3

# 测试自动批次，从 16 开始
python test_direct.py -f 录音.mp3 --ab 16
```

测试完成后，会生成 `test_results_direct.md` 报告文件。

---

## test_direct.py 详细使用指南

### 📋 必须参数

| 参数 | 说明 | 例子 |
|------|------|------|
| `-f` 或 `--file` | 测试用的音频文件 | `-f test.wav` |

### 🎯 测试模式选择（三选一）

#### 模式 1：固定批次范围测试（默认）

```bash
# 测试批次 1 到 10（默认）
python test_direct.py -f 录音.mp3

# 测试批次 5 到 20
python test_direct.py -f 录音.mp3 --min-bs 5 --max-bs 20
```

#### 模式 2：单个自动批次测试

```bash
# 测试从 16 开始的自动批次
python test_direct.py -f 录音.mp3 --ab 16
```

#### 模式 3：多个自动批次测试

```bash
# 测试从 8、16、32 开始的自动批次
python test_direct.py -f 录音.mp3 --auto-bs 8,16,32
```

### ⚙️ 其他常用参数

| 参数 | 说明 | 默认值 | 例子 |
|------|------|--------|------|
| `--repeat` | 每个设置测试多少次 | 10 | `--repeat 5` |
| `--abr` | 自动批次的显存目标 | 0.8-0.9 | `--abr 0.75-0.95` |
| `--min-success` | 最少成功次数才算有效 | 1 | `--min-success 3` |
| `--order` | 测试顺序（random/sequential） | random | `--order sequential` |
| `--resume` | 从上次中断处继续 | - | `--resume` |
| `--no-checkpoint` | 不保存检查点 | - | `--no-checkpoint` |
| `--cooldown` | 两次测试之间冷却秒数 | 2.0 | `--cooldown 3` |
| `-o` | 输出报告文件名 | test_results_direct.md | `-o my_report.md` |

### 🎛️ 功能开关参数

和 `cli.py` 一样：

| 参数 | 说明 |
|------|------|
| `--nfp` | 禁用 FP16 |
| `--pu` | 启用标点预测 |
| `--ts` | 输出时间戳 |

### 📊 完整测试例子

```bash
# 完整示例：测试自动批次 8,16,32，每个测 5 次，启用标点
python test_direct.py -f 录音.mp3 --auto-bs 8,16,32 --repeat 5 --pu
```

---

## 常见问题解答

### Q1: 报错说「显存不足」怎么办？

**A:** 减小批次大小：
```bash
# 原来是 --ab 32，改成 --ab 16 或 --ab 8
python cli.py -f 录音.mp3 --ab 8
```

### Q2: 支持哪些音频/视频格式？

**A:** 支持的格式有：
- 音频：`.wav`, `.mp3`, `.flac`, `.aac`, `.ogg`, `.m4a`, `.wma`
- 视频：`.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.webm`

### Q3: 识别结果保存在哪里？

**A:** 默认在 `results` 文件夹里，也可以用 `-o` 自定义。

### Q4: 怎么中断正在运行的程序？

**A:** 按 `Ctrl + C`（按住 Ctrl 键再按 C）。

### Q5: 输入的文件路径有空格怎么办？

**A:** 用引号把路径括起来：
```bash
python cli.py -f "d:\我的录音\第一集.mp3" --ab 16
```

---

## 完整示例参考

### 示例 1：处理单个文件，自动批次，带标点

```bash
python cli.py -f interview.mp3 --ab 16 --pu
```

### 示例 2：处理整个文件夹，递归，时间戳，自定义输出

```bash
python cli.py -d "d:\会议录音" -r --ab 16 --ts -o "d:\会议转录"
```

### 示例 3：处理多个文件，安静模式

```bash
python cli.py -f audio1.wav audio2.mp3 video.mp4 --ab 16 -q
```

### 示例 4：简单性能测试

```bash
python test_direct.py -f test.wav --ab 16 --repeat 5
```

### 示例 5：完整性能测试

```bash
python test_direct.py -f test.wav --auto-bs 8,16,32 --repeat 10 --pu -o performance_report.md
```

---

## 最后提示

1. **先看帮助：** 任何时候 `--help` 都能救急
2. **从小开始：** 先用小文件测试，没问题再处理大文件
3. **善用复制粘贴：** 在命令行里，选中文字按右键是复制，直接按右键是粘贴
4. **多试几次：** 找到最适合你电脑的参数设置

祝你使用愉快！🎉
