# Leveraging LLM-Generated Explanations for Detecting Emotionally Rewritten Fake News

Code for **Leveraging LLM-Generated Explanations for Detecting Emotionally Rewritten Fake News**. This README describes the paper's **v1** implementation, which incorporates LLM-generated explanations into fake news detection.

## Setup

Use Linux, Python 3.11, and an NVIDIA GPU with CUDA support for training. Run all commands from the repository root.

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
python -m pip install uv
uv sync --locked
```

The lockfile includes PyTorch 2.4.1 and Transformers 4.57.6. Check that PyTorch can access your GPU:

```bash
uv run python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The default backbone is `FacebookAI/roberta-base`, downloaded automatically from Hugging Face on first use. Explanations are provided in the dataset; training does not require an LLM API key.

## Data

We use **PolitiFact**, **GossipCop**, and **LUN**, with original news, generated explanations, and emotion-specific test sets (anger, happiness, joy, and sadness).

Download the released data from [chlorane/Emotion_Fakenews](https://huggingface.co/datasets/chlorane/Emotion_Fakenews):

```bash
uv run hf download chlorane/Emotion_Fakenews --repo-type dataset --local-dir data_download
mkdir -p dataset/emotions
cp -r data_download/news_articles dataset/
cp data_download/emotion/*_test_*.pkl dataset/emotions/
```

The Hugging Face directory is named `emotion`; the code expects `dataset/emotions`. The six `news_articles/*_re.pkl` files and twelve emotion-specific test files match our local experiment inputs byte for byte. The loader reads `news`, `labels`, and `explanation` from the original news files, and pairs rewritten test news with the corresponding original labels and explanations.

Training also requires the `reframings` and `veracity_attributions` files from [SheepDog](https://github.com/jiayingwu19/SheepDog). These are not included in the Hugging Face release above. Obtain them with:

```bash
git clone --depth 1 https://github.com/jiayingwu19/SheepDog.git SheepDog
cp -r SheepDog/data/reframings SheepDog/data/veracity_attributions dataset/
```

The expected layout is:

```text
dataset/
├── news_articles/          # {dataset}_{train,test}_re.pkl
├── emotions/               # {dataset}_test_{emotion}.pkl
├── reframings/             # Training news in four writing styles
└── veracity_attributions/  # Fine-grained supervision from SheepDog
```

## Training

For example, train v1 on PolitiFact for five runs of five epochs each:

```bash
mkdir -p results
CUDA_VISIBLE_DEVICES=0 uv run src/sheepdog.py \
  --dataset_name politifact \
  --model_name sheepdog \
  --model_version v1 \
  --encoder_type roberta \
  --n_epochs 5 \
  --iters 5 \
  --batch_size 4 \
  --use_match_loss \
  --run_name politifact_v1 \
  > results/politifact_v1.log 2>&1
```

Set `--dataset_name` to `gossipcop` or `lun` for the other datasets, and choose a new `--run_name` for each experiment. `--use_match_loss` enables the text–explanation matching loss. Adjust `CUDA_VISIBLE_DEVICES` to select your GPU.

Checkpoints are saved as `checkpoints/<dataset>/<run_name>/iter*.m`; logs and metric summaries are stored under `results/`. Training evaluates the original test set and automatically discovers the available emotion-specific test sets. Use `--test_emotions anger,sadness` to select a subset.

## Evaluation

Evaluate the checkpoints from the example above:

```bash
CUDA_VISIBLE_DEVICES=0 uv run src/eval_checkpoints.py \
  --dataset_name politifact \
  --checkpoint_dir checkpoints/politifact/politifact_v1 \
  --model_version v1 \
  --encoder_type roberta \
  > results/politifact_v1_eval.log 2>&1
```

The evaluator reports accuracy, precision, recall, and F1 on the original and emotion-specific test sets, including means and standard deviations across checkpoints. Alternatively, edit `dataset_name`, `checkpoint_dir`, and `gpu_id` in `test5.sh`, then run `bash test5.sh`.

## Acknowledgments

Our work builds on **SheepDog**, introduced in [Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks](https://arxiv.org/abs/2310.10830). We thank the authors for making their [code and data](https://github.com/jiayingwu19/SheepDog) publicly available.
