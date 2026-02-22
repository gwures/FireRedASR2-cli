<div align="center">
<h1>
FireRedASR
<br>
工业级全功能语音识别系统
</h1>
</div>

[[论文]](https://arxiv.org/pdf/2501.14350)
[[模型]](https://huggingface.co/FireRedTeam)
[[博客]](https://fireredteam.github.io/demos/firered_asr/)
[[演示]](https://huggingface.co/spaces/FireRedTeam/FireRedASR)

## 项目概述

FireRedASR 是一个基于 FireRedASR2S 构建的先进工业级全功能语音识别系统，集成了 ASR（自动语音识别）、VAD（语音活动检测）和标点符号预测等多个模块，所有模块均达到业界领先水平。本项目提供了命令行接口和自动化批次处理功能，方便用户快速高效地进行语音转文字任务。

### 核心特性

- **高精度识别**：基于 FireRedASR2-AED，支持中文（普通话及 20+ 方言/口音）、英文、码混合和歌词识别
- **自动化批次处理**：智能显存管理，自动调整批次大小以优化性能
- **多功能集成**：VAD + ASR + 标点预测一站式处理
- **多格式支持**：支持 WAV、MP3、FLAC、AAC、OGG、M4A、WMA、MP4、AVI、MKV、MOV、WMV、WebM 等多种音频/视频格式
- **时间戳输出**：支持词级和句级时间戳，方便生成字幕文件（SRT）
- **GPU 加速**：支持 FP16 精度，大幅提升推理速度

---

## 核心功能介绍

### 1. 自动语音识别 (ASR)
- **FireRedASR2-AED**：基于注意力机制的编码器-解码器架构
- 普通话平均 CER：3.05%
- 中文方言平均 CER：11.67%
- 支持词级时间戳和置信度分数

### 2. 语音活动检测 (VAD)
- **FireRedVAD**：支持 100+ 种语言的语音/歌声/音乐检测
- F1 分数：97.57%
- 支持非流式和流式 VAD

### 3. 标点符号预测
- **FireRedPunc**：支持中英文标点符号预测
- 平均 F1 分数：78.90%

### 4. 智能批次管理
- 自动监控显存利用率
- 动态调整 ASR 批次大小
- OOM（内存不足）自动恢复机制

---

## 环境要求

### 系统要求
- **操作系统**：Windows / Linux / macOS
- **Python 版本**：3.10 及以上
- **GPU（推荐）**：支持 CUDA 的 NVIDIA GPU（推荐 8GB 显存以上）
- **CPU（仅推理）**：4 核以上，8GB 内存

### 依赖库
- numpy == 1.26.1
- modelscope >= 1.14.0
- transformers == 5.1.0
- cn2an == 0.5.23
- kaldiio == 2.18.0
- kaldi_native_fbank >= 1.15
- sentencepiece == 0.1.99
- soundfile == 0.12.1
- textgrid
- setuptools
- nvidia-ml-py

---

## 安装步骤

### 步骤 1：克隆或下载项目
```bash
git clone https://github.com/your-username/fireredasr.git
cd fireredasr
```


### 步骤 2：安装依赖及pytorch
```bash
pip install -r requirements.txt
```
必须从https://pytorch.org/get-started/locally/  安装对应自己cuda版本的pytorch。
### 步骤 3：下载预训练模型
项目提供了便捷的模型下载脚本：

```bash
python download_models.py
```

或者手动下载：

#### 通过 ModelScope（推荐中国大陆用户）
```bash
pip install -U modelscope
modelscope download --model xukaituo/FireRedASR2-AED --local_dir ./pretrained_models/FireRedASR2-AED
modelscope download --model xukaituo/FireRedVAD --local_dir ./pretrained_models/FireRedVAD
modelscope download --model xukaituo/FireRedPunc --local_dir ./pretrained_models/FireRedPunc
```

#### 通过 Hugging Face
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download FireRedTeam/FireRedASR2-AED --local-dir ./pretrained_models/FireRedASR2-AED
huggingface-cli download FireRedTeam/FireRedVAD --local-dir ./pretrained_models/FireRedVAD
huggingface-cli download FireRedTeam/FireRedPunc --local-dir ./pretrained_models/FireRedPunc
```

---

## 使用指南

### 基本操作

#### 1. 查看帮助信息
```bash
python cli.py --help
```

#### 2. 处理单个音频文件（自动批次）
```bash
python cli.py -f audio.mp3 --ab 16
```

#### 3. 处理单个音频文件（固定批次）
```bash
python cli.py -f audio.mp3 --bs 8
```
简单地说，必须使用自动批次（--ab）或者固定批次（--bs）参数，其余参数如果不指定将使用默认参数（--fp，--npu，--nts），即默认以fp16精度、不加标点、不返回词级时间戳的方式执行，这是效率最高的模式（显存占用近3G）。
不返回词级时间戳完全不影响字幕的生成，字幕也是同步的，只是缺少词句精度，导致字幕的某些片段可能显示过长或过短。
关于是否需要开启标点，punc模型占用显存400M左右，速度极快，几秒内就能完成加标点，可以按需开启。如果以全精度、加标点运行，显存占用近6G。
注意不建议手动修改config.py中的配置模板，因为实际上是按cli传递的参数运行的。

#### 4. 处理多个文件
```bash
python cli.py -f file1.mp3 file2.wav file3.flac --ab 16
```

#### 5. 处理整个目录
```bash
python cli.py -d /path/to/audio --ab 16
```

#### 6. 递归处理目录（包含子目录）
```bash
python cli.py -d /path/to/audio -r --ab 16
```

### 高级功能

#### 启用标点符号预测
```bash
python cli.py -f audio.mp3 --ab 16 --pu
```

#### 输出时间戳
```bash
python cli.py -f audio.mp3 --ab 16 --ts
```

#### 禁用 FP16（使用 FP32 精度）
```bash
python cli.py -f audio.mp3 --ab 16 --nfp
```

#### 自定义显存目标区间
```bash
python cli.py -f audio.mp3 --ab 16 --abr 0.75-0.95
```

#### 自定义输出目录
```bash
python cli.py -f audio.mp3 --ab 16 -o my_output
```

#### 静默模式（仅显示进度和错误）
```bash
python cli.py -f audio.mp3 --ab 16 -q
```

#### 详细日志模式
```bash
python cli.py -f audio.mp3 --ab 16 -v
```

### 输出说明

处理完成后，结果将保存在 `results` 目录（或您指定的目录）中，每个输入文件会生成两个输出文件：

1. **<文件名>.srt**：字幕文件
2. **<文件名>.txt**：纯文本转写结果


## 常见问题解答 (FAQ)

### Q1: 支持哪些音频格式？
A: 支持 WAV、MP3、FLAC、AAC、OGG、M4A、WMA 等音频格式，以及 MP4、AVI、MKV、MOV、WMV、WebM 等视频格式。系统会自动将输入转换为 16kHz 16-bit 单声道 PCM 格式。

### Q2: ASR 模型对输入音频长度有什么限制？
A: FireRedASR2-AED 支持最长约 60 秒的音频输入。超过 60 秒可能会产生幻觉问题，超过 200 秒会触发位置编码错误。好在本项目的VAD 会自动将长音频切分为合适的片段。

### Q3: 如何选择批次大小？
A: 首次使用，推荐使用 `--ab 3` 参数启用自动批次管理，系统会根据显存利用率自动调整。然后根据结果后续，使用 `--ab N` 固定批次 `--bs N`。
批次越大，并不意味着转写效率越高，即便显存足够富裕。即批次大小与RTF并非是线性的反比关系，甚至可能产生反效果。效率根在GPU的计算能力而不显存大小。
以本人RTX3060 6G显存跑默认模式，bs 1到bs 10，各10次，总计100次的测试，RTF在0.1523-0.2902之间，似乎固定批次为3-5之间效果最佳。如果以全精度、加标点运行，RTF可能在0.4-0.5，此项我没做大量测试.
如需测试，请参考**cli-help.md**，运行python test_direct.py，注意这个测试命令后面的文件必须是wav格式（否则会报错，如此设计是为了避免同一个文件运行ffmpeg多次），请先手动完成格式转换（ffmpeg -i <input_audio_path> -ar 16000 -ac 1 -acodec pcm_s16le -f wav <output_wav_path>）。该测试命令，如果不额外指定repeat参数，将默认循环测试10次。


### Q4: 出现 CUDA out of memory 错误怎么办？
A: 
1. 使用自动批次模式 `--ab`，系统会自动处理 OOM
2. 减小初始批次大小，如 `--ab 4`
3. 启用 FP16（默认已启用）
4. 关闭其他占用显存的程序

### Q5: 如何提高识别准确率？
A:
1. 确保音频质量良好，信噪比高
2. 避免背景噪音过大
3. 对于特定领域，可以考虑微调模型
4. 启用标点预测 `--pu` 提升可读性

### 	Q6:是否支持triton_tensorrt推理？
A：尽管TensorRT-LLM效率最高，但原项目runtime/triton_tensorrt/pyproject.toml指定某些依赖并没有windows编译版本，最佳实践是docker部署，故而本项目没有添加这部分，还需要额外安装docker。
### Q7：为什是命令行模式而不是图形化交互？
A:本项目前期原来是用NiceGUI写的前端，光UI设计和持久化，修复一系列前端bug就花了一周时间，最后还是选择壮士断腕，移除了所有前端代码。

##⚠️
1.如遇到单个任务转写出错，可修改punc.py第295行“ assert token == timestamp[0], f"{token}/{timestamp}为
“if token != timestamp[0]:
                    logger.warning(f"Token mismatch: reconstructed='{token}' vs original='{timestamp[0]}' - using original")
                    token = timestamp[0] if timestamp[0] else token”
2.相比于原开源项目，本项目最核心的变化是移除了整个FireRedLID模块，也不下载该模型，因为于转写任务无用。同时使用更激进的 torch.inference_mode()而不是原来的torch.no_grad() 进行AED、PUNC的推理，同时由于VAD并未启用GPU，VAD保留了torch.no_grad()。

## 贡献指南

我们欢迎社区贡献！以下是参与项目的方式：

### 报告问题
- 使用 GitHub Issues 报告 bug
- 详细描述复现步骤
- 提供系统环境和错误日志



## 开源许可

本项目基于以下开源项目：

- FireRedASR2S：https://github.com/FireRedTeam/FireRedASR2S
- Qwen
- WenetSpeech-Yue
- WenetSpeech-Chuan


## 致谢

感谢 FireRedTeam 开源 FireRedASR2S，以及所有为本项目做出贡献的开发者。

