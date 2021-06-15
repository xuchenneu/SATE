# Fairseq_ST使用说明

# 简要说明

Fairseq_ST基于原始的Fairseq，提高了程序易用性以及对语音到文本任务的适配。

目前支持功能：

- 针对每个数据集创建egs文件夹保存运行脚本，目前包括LibriSpeech语音识别数据集和MuST-C语音翻译数据集
- 通过读取yaml配置文件进行训练
- 支持ctc多任务学习
- 使用ST相似的流程训练MT模型（在线分词）
- 速度扰动 (需要torchaudio ≥ 0.8.0)
- MT pipeline(bin)
- Conformer模型结构
- 预训练模型加载
- 相对位置表示
- SATE模型结构

# 需求条件

1. Python ≥3.6
2. torch ≥ 1.4, torchaudio ≥ 0.4.0, cuda ≥ 10.1
3. apex
```
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
4. nccl
```
make -j src.build CUDA_HOME=<path to cuda install>
```
5. gcc ≥ 4.9
6. python包 pandas sentencepiece configargparse gpustat tensorboard editdistance

# 代码结构

此外，语音翻译任务需要对每个任务预先下载好原始数据，除了已经提供的数据集，如LibriSpeech和MuST-C外，其他数据集需要额外编写代码进行处理，参考examples/speech_to_text路径下的处理文件。

运行脚本存放于fairseq根目录下的egs文件夹，针对每个数据集分别建立了不同的文件夹来执行操作，目前包括语音识别数据集LibriSpeech以及语音翻译数据集MuST-C的执行脚本。

以librispeech文件夹举例，其中包含以下文件：

```markdown
librispeech
├── conf
│   └── train_config.yaml
├── local
│   ├── monitor.sh
│   ├── parse_options.sh
│   ├── path.sh
│   └── utils.sh
├── decode.sh
├── history.log
├── run.sh
├── train_history.log
└── train.sh
```

- run.sh是核心脚本，包含了数据的处理以及模型的训练及解码，train.sh和decode.sh分别调用run.sh来实现单独的训练和解码功能。
- history.log保存了历史的训练信息，包含模型训练使用的显卡、数据集以及模型存储位置。
- conf文件夹下为训练配置，目前修改了Fairseq使其支持读取yaml配置文件。模型训练所要使用的配置可以在该文件中进行设置。
- local文件夹下为一些常用脚本
    - monitor.sh为检测程序，可以检测是否有显卡空闲，如果空闲一定数据，则执行某个任务
    - parse_options.sh为支持其他文件调用run.sh的辅助文件
    - path.sh暂时还未使用
    - utils.sh中包含了显卡检测函数

mustc文件夹和librispeech文件夹类似，其中run.sh额外支持了语音翻译任务的训练。
