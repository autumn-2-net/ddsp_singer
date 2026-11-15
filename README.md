# ddsp singer
# torch>2.1

NSF cgMLP Vocos 骨干使用独立配置 `configs/nsf_cgmlp_vocos.yaml`。链路为
FastSpeech2 → cgMLP + Combtooth 激励 → Vocos iSTFT；F0 只进入激励源，
不构造 pitch embedding，也不使用旧 Pencoder/wnd。`nsf_cgmlp_vocos.py`
接收的 conditioning 在这里是 FastSpeech2 隐藏特征，而不是目标 mel。
只保留 FS2 和 mel2ph 帧展开，其后的所有旧网络都由 NSF cgMLP Vocos 替换。
新分支使用波形 L1 + 多尺度 STFT loss（频谱收敛项 + log 幅度项），
由 `lambda_l1`、`lambda_mstft` 控制权重，默认均为 1.0；暂不包含鉴别器。

在仓库根目录使用 PowerShell 启动训练：

```powershell
& 'C:\Users\autumn\.conda\envs\py10\python.exe' run_train.py --config configs/nsf_cgmlp_vocos.yaml
```

配置不继承或修改旧配置；默认数据索引为 `datax/`、字典为 `dictionary.txt`。
模型输出到 `ckpt/nsf_cgmlp_vocos`，
并保存合并后的 `config.yaml`。可通过 `--work-dir` 更改目录，
通过 `--resume <checkpoint>` 恢复训练。新骨干需要重新训练；
`--init-checkpoint` 只接受架构相同的 NSF 模型，不会自动加载旧 DDSP 权重。
未设置 `backbone` 的原配置仍使用 DDSP。

训练后直接生成波形：

```powershell
& 'C:\Users\autumn\.conda\envs\py10\python.exe' infer_nsf_vocos.py song.ds --config ckpt/nsf_cgmlp_vocos/config.yaml --checkpoint ckpt/nsf_cgmlp_vocos/model_ckpt_steps_7999.ckpt --output output/song.wav
```

`--transpose` 以半音调整 F0；支持 `.ds` 分段 offset，无 offset 时顺序拼接。
不需要另外加载 NSF-HiFiGAN。连续 F0 的训练配置使用 `interp_uv: true`。

小型前反向及 F0 路径检查（有 CUDA 时包含 BF16）：

```powershell
& 'C:\Users\autumn\.conda\envs\py10\python.exe' -m unittest discover -s tests -v
```

多级版本使用独立配置 `configs/nsf_cgmlp_vocos_multistage.yaml`：

| 级别 | 帧数 | 通道数 |
| --- | --- | --- |
| front 调制层／第一组残差块 | T | 768 |
| 第一次通道扩展 | T | 1024 |
| 第二次通道扩展 | T | 2048 |

先由独立的 `front cgMLP → filter_head` 调制 NSF，源频谱只计算一次。
随后每级在 cgMLP 残差块之前，将同一份滤波后的 NSF 投影加入残差流。
最终输出头将主干特征与这份 NSF 频谱的实部／虚部拼接，再通过 linear
预测幅度／相位。`num_front` 控制前置调制层数，`num_blocks` 控制后续各级层数。
默认输出头输入为 2048 + 1026 = 3074 维，输出为 1026 维。
各级只扩展通道维度，不做时间上采样。使用 FFT=1024、hop=512 的 iSTFT
输出 T×512 个采样点；
最终频谱没有额外的 NSF 直连。损失仍为波形 L1 + MSTFT，TensorBoard
会额外记录 `loss_l1` 和 `loss_mstft`，便于区分损失来源。

顶层 `fft_size` 统一控制 NSF 源分析、调制和最终合成，没有逐层 FFT 配置。
测试 FFT=2048 时将 `fft_size` 和数据分析的 `win_size` 一起设为 2048，
hop 仍为 512。这会改变滤波器、NSF 投影和输出头形状，需要新的训练目录。

多级模型需要从头训练，参数形状与单级模型不兼容。默认 batch 为 1，
逐级加宽会增加显存需求。

```powershell
& 'C:\Users\autumn\.conda\envs\py10\python.exe' run_train.py --config configs/nsf_cgmlp_vocos_multistage.yaml --work-dir ckpt/nsf_cgmlp_vocos_multistage
```

```powershell
& 'C:\Users\autumn\.conda\envs\py10\python.exe' -m tensorboard.main --logdir ckpt/nsf_cgmlp_vocos_multistage/lightning_logs --port 6006
```
