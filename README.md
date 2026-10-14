# DarkStream: Real-Time Speech Anonymization with Low Latency (ASRU 2025)

[Waris Quamer](https://github.com/warisqr007) and [Ricardo Gutierrez-Osuna](https://scholar.google.com/citations?user=UnuQfEwAAAAJ&hl=en)


<a href="https://arxiv.org/abs/2509.04667">
  <img src="https://img.shields.io/badge/arXiv-2509.04667-%23B31B1B">
</a>
<!-- <a href="">
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-FFD21E">
</a> -->
<a href="https://warisqr007.github.io/demos/darkstream/">
  <img src="https://img.shields.io/badge/Demo%20Page-online-brightgreen">
</a>
<br>

## Abstract

We propose DarkStream, a streaming speech synthesis model for real-time speaker anonymization.
To improve content encoding under strict latency constraints, DarkStream combines a causal waveform
encoder, a short lookahead buffer, and transformer-based contextual layers. To further reduce
inference time, the model generates waveforms directly via a neural vocoder, thus removing
intermediate mel-spectrogram conversions. Finally, DarkStream anonymizes speaker identity by
injecting a GAN-generated pseudo-speaker embedding into linguistic features from the content encoder.
Evaluations show our model achieves strong anonymization, yielding close to 50% speaker verification
EER (near-chance performance) on the lazy-informed attack scenario, while maintaining acceptable
linguistic intelligibility (WER within 9%). By balancing low-latency, robust privacy, and minimal
intelligibility degradation, DarkStream provides a practical solution for privacy-preserving
real-time speech communication.

## Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Inference](#inference)
- [Training](#training)
- [SLT24 Model](#slt24-model)

## Installation

```bash
git clone https://github.com/PSI-TAMU/DarkStream.git
cd DarkStream

conda create -n darkstream python=3.10
conda activate darkstream

conda install -c conda-forge git-lfs -y
pip install -r requirements.txt
```

## Data Preparation

DarkStream uses the same data pipeline as [StreamVC](https://github.com/PSI-TAMU/streamVC). Follow the preprocessing instructions in that repository.

## Pretrained Checkpoints

### Model naming

| Term | Meaning |
|------|---------|
| **Wav** | Causal waveform encoder: the content encoder takes raw audio directly, with no mel-spectrogram front end. |
| **CL** | Contextual layers: a causal transformer stacked on top of the waveform encoder. |
| **KM** | Content features are quantized with a k-means model (256 clusters) before decoding. |
| **ResidualKM (N levels)** | Residual k-means quantization with N stages, each quantizing the residual of the previous stage. |
| **w/ logits** | The decoder receives the content encoder's output logits (through a linear projector) instead of its hidden features. |
| **Lookahead** | Future context available to the encoder. With 20 ms frames, 0 / 140 / 280 ms correspond to 0 / 7 / 14 frames (`nola` / `la7` / `la14` in file and config names). |

In all variants, the content encoder is first pretrained to predict HuBERT k-means units (200 clusters) and is then frozen while the decoder is trained for resynthesis.

### DarkStream (Wav + CL)

A HiFi-GAN-style causal convolutional encoder and decoder (implemented with SEANet blocks from Mimi), with a causal transformer as the contextual layer.

#### Paper models

The configurations evaluated in the DarkStream paper. ⭐ marks the main DarkStream model (140 ms lookahead with k-means quantization), which gives the anonymization results reported in the abstract (≈47% lazy-informed EER, ≈9.5% WER).

| Model | Lookahead | Module | Download |
|-------|-----------|--------|----------|
| Wav + CL | 0 ms   | `MimiResynthesisModule` | [Google Drive](https://drive.google.com/file/d/1JhtroQHxV-1qTzETtxidXE4sg5KznplQ/view?usp=sharing) |
| Wav + CL | 140 ms | `MimiResynthesisModule` | [Google Drive](https://drive.google.com/file/d/1pohNzyyvuNaJu4Nq44SWG5iEpxJODSsF/view?usp=sharing) |
| Wav + CL | 280 ms | `MimiResynthesisModule` | [Google Drive](https://drive.google.com/file/d/1-hj2MwraXHFoouK2l2YfmRibFl6krZmf/view?usp=sharing) |
| ⭐ **Wav + CL + KM (DarkStream)** | **140 ms** | `MimiKMResynthesisModule` | [Google Drive](https://drive.google.com/file/d/1pfdCHRAacD3ppdmx3B8IsMmloh6p02Xm/view?usp=sharing) |

#### Additional variants (not in the paper)

Extensions released with this repository that were not evaluated in the paper.

| Model | Lookahead | Module | Download |
|-------|-----------|--------|----------|
| Wav + CL + ResidualKM (2 levels) | 140 ms | `MimiResidualKMResynthesisModule` | [Google Drive](https://drive.google.com/file/d/17ki6NGtJLxmSbJzAmGZlVH8AyirAwdvw/view?usp=sharing) |
| Wav + CL + ResidualKM (4 levels) | 140 ms | `MimiResidualKMResynthesisModule` | [Google Drive](https://drive.google.com/file/d/12ij4ybFNmo8WLEBMoRzC9j88YIm5PGNh/view?usp=sharing) |
| Wav + CL + ResidualKM (8 levels) | 140 ms | `MimiResidualKMResynthesisModule` | [Google Drive](https://drive.google.com/file/d/1kJCgcYM06Xlxkj8Y_akSEpwKu7J7VVFe/view?usp=sharing) |
| Wav + CL (w/ logits) | 140 ms | `MimiResynthesisWithLogitsModule` | [Google Drive](https://drive.google.com/file/d/1pyByzdhqmoWmMKNuqS8N841KhueJw0-6/view?usp=sharing) |

Place downloaded checkpoints in [`ckpts/`](ckpts). The k-means models used by the KM variants (`km256_model.bin`, `km256_centers.pth`) are already included there.

More pretrained checkpoints, including the stage-1 content encoders (e.g. `mimipasr_nola.ckpt`) used as `encoder_path` when training the resynthesis models, are available in this [Google Drive folder](https://drive.google.com/drive/folders/1pd8eD5kuMcou8_P1L9NS2ZoOet8ScGZy).

## Inference

See [`notebooks/inference_darkstream.ipynb`](notebooks/inference_darkstream.ipynb) for non-streaming synthesis, frame-by-frame streaming synthesis, and anonymization with a GAN-generated pseudo-speaker embedding. Select the variant with `model_type`:

| `model_type` | Model |
|--------------|-------|
| `darkstream_without_kmeans` | Wav + CL |
| `darkstream` | Wav + CL + KM |
| `darkstream_with_residualkmeans` | Wav + CL + ResidualKM |
| `darkstream_with_logits` | Wav + CL (w/ logits) |

Sample utterances for testing are provided in [`sample_wav/`](sample_wav).

## Training

Training uses [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html); each run combines a data config and a model config:

```bash
python src/main.py fit -c configs/data/<data>.yaml -c configs/model/<model>.yaml --trainer.logger.name <run_name>

# Resume from a checkpoint
python src/main.py fit -c configs/data/<data>.yaml -c configs/model/<model>.yaml --ckpt_path <ckpt_path> --trainer.logger.id <run_id>
```

DarkStream is trained in four stages. The commands below use the 140 ms (`la7`) configs.

**1. Content encoder.** Train the encoder to predict HuBERT units.

```bash
python src/main.py fit -c configs/data/pasr.yaml -c configs/model/mimipasr_la7.yaml
```

To change the lookahead, set `model.init_args.seanet_args.look_ahead_size` (in 20 ms frames).

**2. Resynthesis model.** Set `model.init_args.encoder_path` in the model config to the checkpoint from stage 1 (or a pretrained content encoder from the [Google Drive folder](https://drive.google.com/drive/folders/1pd8eD5kuMcou8_P1L9NS2ZoOet8ScGZy)).

```bash
python src/main.py fit -c configs/data/resynthesis.yaml -c configs/model/mimiresynthesis_la7.yaml
```

Use `mimiresynthesis_nola.yaml` or `mimiresynthesis_la14.yaml` for the 0 ms and 280 ms variants, and `mimiresynthesiswithlogits_la7.yaml` for the logits variant.

**3. K-means model.** Extract content features with [`preprocess/dump_content_features.py`](preprocess/dump_content_features.py) (edit the checkpoint path inside to match your model), then train k-means with [`src/components/quantizer/learn_kmeans.py`](src/components/quantizer/learn_kmeans.py).

**4. (Optional) Fine-tune with quantized features.** In the model config set:

- `model.init_args.encoder_path`: content encoder checkpoint from stage 1
- `model.init_args.kmeans_path`: k-means model from stage 3
- `model.init_args.decoder_pretrained: true`
- `model.init_args.decoder_path`: resynthesis checkpoint from stage 2

```bash
python src/main.py fit -c configs/data/resynthesis.yaml -c configs/model/mimikmresynthesis_la7.yaml
```

For codebase layout, configuration conventions, and environment tips, see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## SLT24 Model

This repository also provides checkpoints for our earlier SLT 2024 architecture, referred to as **Wav** in the comparisons above. It uses a HiFi-GAN-style causal encoder and decoder without contextual layers, and is loaded with `ResynthesisModule`.

| Model | Lookahead | Download |
|-------|-----------|----------|
| Wav | 0 ms   | [Google Drive](https://drive.google.com/file/d/1YeGO6_NhyGbw-zFiYe04jJDu8UFEJq1e/view?usp=sharing) |
| Wav | 140 ms | [Google Drive](https://drive.google.com/file/d/1k3WrFjINpW8ggR6v0aDx8geCWrQYmmoh/view?usp=sharing) |
| Wav | 280 ms | [Google Drive](https://drive.google.com/file/d/10kBxii3ragO_a4nvt6syCvVI-tKESJfd/view?usp=sharing) |

See [`notebooks/inference_slt24.ipynb`](notebooks/inference_slt24.ipynb) for usage. For more details on this model, see the [StreamVC repository](https://github.com/PSI-TAMU/streamVC).

## Citation
```
@INPROCEEDINGS{quamer2025darkstreamrealtimespeechanonymization,
  author={Quamer, Waris and Gutierrez-Osuna, Ricardo},
  booktitle={2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)}, 
  title={DarkStream: real-time speech anonymization with low latency}, 
  year={2025},
  volume={},
  number={},
  pages={1-7},
  keywords={Data privacy;Privacy;Vocoders;Linguistics;Transformers;Real-time systems;Information filtering;Speech synthesis;Low latency communication;Information integrity;speech synthesis;speaker anonymization;voice conversion},
  doi={10.1109/ASRU65441.2025.11434609}
}
```

## License

This project is released under the [CC0 1.0 Universal](LICENSE) license.
