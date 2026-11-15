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
