import argparse
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertModel as HFBertModel

sys.path.append(os.getcwd())
from roberta_encoder import RobertaEncoderResults
from utils.load_data import load_articles, load_emotion_tests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="politifact", type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str,
                        help="Checkpoint folder, e.g., checkpoints/gossipcop/20260407_xxx")
    parser.add_argument("--model_version", default="v1", type=str, choices=["v1", "v2", "v3", "v4", "v5"])
    parser.add_argument("--encoder_type", default="roberta", type=str, choices=["roberta", "bert"])
    parser.add_argument("--fusion_layer", default=11, type=int)
    parser.add_argument("--dim_common", default=256, type=int)
    parser.add_argument("--n_attn_heads", default=1, type=int)
    parser.add_argument("--disable_gate", action="store_true")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--test_emotions", default="",
                        help="Comma-separated emotions; empty means auto-detect")
    parser.add_argument("--use_sliding_window", action="store_true")
    parser.add_argument("--window_size", default=512, type=int)
    parser.add_argument("--window_stride", default=256, type=int)
    parser.add_argument("--eval_max_windows", default=32, type=int)
    return parser.parse_args()


class NewsDataset(Dataset):
    def __init__(self, texts, labels, explanation, tokenizer, exp_tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.explanation = explanation
        self.tokenizer = tokenizer
        self.exp_tokenizer = exp_tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.texts[item]
        explanation = self.explanation[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        exp_encoding = self.exp_tokenizer.encode_plus(
            explanation,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "news_text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "explanation": explanation,
            "input_ids_exp": exp_encoding["input_ids"].flatten(),
            "attention_mask_exp": exp_encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.texts)


class NewsDatasetConcatenated(Dataset):
    def __init__(self, texts, labels, explanation, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.explanation = explanation
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.texts[item]
        explanation = self.explanation[item]
        label = self.labels[item]

        text_tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=256)
        exp_tokens = self.tokenizer.encode(explanation, add_special_tokens=False, truncation=True, max_length=256)

        if len(text_tokens) + len(exp_tokens) > 509:
            text_tokens = text_tokens[:254]
            exp_tokens = exp_tokens[:254]

        input_ids = [self.tokenizer.cls_token_id] + text_tokens + [self.tokenizer.sep_token_id] + exp_tokens + [self.tokenizer.sep_token_id]
        text_boundary = len(text_tokens) + 2

        attention_mask = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))

        return {
            "news_text": text,
            "explanation": explanation,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "text_boundary": text_boundary,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.texts)


class TransformerClassifier(nn.Module):
    def __init__(self, args, n_classes, backbone_cls, pretrained_name):
        super().__init__()
        self.model_version = args.model_version
        self.backbone = backbone_cls.from_pretrained(
            pretrained_name,
            fusion_layer=args.fusion_layer,
            dim_common=args.dim_common,
            n_attn_heads=args.n_attn_heads,
            disable_gate=args.disable_gate,
        )
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(p=0.5)
        self.fc_out = nn.Linear(hidden_size, n_classes)
        self.binary_transform = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, exp_feature=None, exp_attention_mask=None, text_boundary=None):
        if self.model_version == "v4":
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, text_boundary=text_boundary)
        elif self.model_version == "v2":
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                exp_feature=exp_feature,
                exp_attention_mask=exp_attention_mask,
            )
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, exp_feature=exp_feature)

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None and isinstance(outputs, (tuple, list)):
            pooled = outputs[1]
        pooled_outputs = self.dropout(pooled)
        output = self.fc_out(pooled_outputs)
        binary_output = self.binary_transform(pooled_outputs)
        return output, binary_output


def build_concat_ids(tokenizer, text, explanation, max_len=512):
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    exp_tokens = tokenizer.encode(explanation, add_special_tokens=False)
    max_total = max_len - 3
    if len(text_tokens) + len(exp_tokens) > max_total:
        t_keep = max(1, int(max_total * len(text_tokens) / (len(text_tokens) + len(exp_tokens))))
        e_keep = max_total - t_keep
        text_tokens = text_tokens[:t_keep]
        exp_tokens = exp_tokens[:e_keep]
    input_ids = [tokenizer.cls_token_id] + text_tokens + [tokenizer.sep_token_id] + exp_tokens + [tokenizer.sep_token_id]
    text_boundary = len(text_tokens) + 2
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        text_boundary,
    )


def sliding_window_infer_v4(model, tokenizer, text, explanation, window_size=512, stride=256, device="cuda", max_windows=None):
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    exp_tokens = tokenizer.encode(explanation, add_special_tokens=False)
    full_ids = [tokenizer.cls_token_id] + text_tokens + [tokenizer.sep_token_id] + exp_tokens + [tokenizer.sep_token_id]
    boundary = len(text_tokens) + 2
    total_len = len(full_ids)

    if total_len <= window_size:
        ids, mask, tb = build_concat_ids(tokenizer, text, explanation, max_len=window_size)
        ids = ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        _, out_bi = model(input_ids=ids, attention_mask=mask, text_boundary=tb)
        return out_bi

    logits_bi_list = []
    starts = list(range(0, max(1, total_len - window_size + 1), stride))
    if max_windows is not None and len(starts) > max_windows:
        idxs = np.linspace(0, len(starts) - 1, num=max_windows, dtype=int)
        starts = [starts[i] for i in idxs]

    for start in starts:
        end = min(start + window_size, total_len)
        window_ids = full_ids[start:end]
        tb_win = boundary - start
        tb_win = int(max(0, min(tb_win, len(window_ids))))

        pad_len = window_size - len(window_ids)
        input_ids = window_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(window_ids) + [0] * pad_len

        ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            _, out_bi = model(input_ids=ids, attention_mask=mask, text_boundary=tb_win)
        logits_bi_list.append(out_bi)

    logits_bi = torch.mean(torch.stack(logits_bi_list, dim=0), dim=0)
    return logits_bi


def create_eval_loader(args, contents, labels, explanation, tokenizer, exp_tokenizer):
    if args.model_version in ["v4", "v5"]:
        ds = NewsDatasetConcatenated(
            texts=contents,
            labels=np.array(labels),
            explanation=explanation,
            tokenizer=tokenizer,
            max_len=args.max_len,
        )
    else:
        ds = NewsDataset(
            texts=contents,
            labels=np.array(labels),
            explanation=explanation,
            tokenizer=tokenizer,
            exp_tokenizer=exp_tokenizer,
            max_len=args.max_len,
        )
    return DataLoader(ds, batch_size=args.batch_size, num_workers=0)


def build_backbone(args):
    model_registry = {
        "v1": ("modeling_roberta", "v1 (multi-head attention)"),
        "v2": ("modeling_roberta2", "v2 (token-wise attention)"),
        "v3": ("modeling_roberta3", "v3 (gated+mamba)"),
        "v4": ("modeling_roberta4", "v4 (concatenation+gate+decoder)"),
        "v5": ("modeling_roberta5", "v5 (enhanced method 2 with improved loss)"),
    }
    module_name, version_label = model_registry[args.model_version]
    model_module = importlib.import_module(module_name)

    backbone_options = {"roberta": getattr(model_module, "RobertaModel")}
    if hasattr(model_module, "BertModel"):
        backbone_options["bert"] = getattr(model_module, "BertModel")

    if args.encoder_type not in backbone_options:
        raise ValueError(f"Encoder type '{args.encoder_type}' is not supported for model version '{args.model_version}'.")

    return backbone_options[args.encoder_type], version_label


def parse_iter_idx(path: Path):
    match = re.search(r"iter(\d+)\.m$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def collect_checkpoints(checkpoint_dir: Path):
    all_ckpts = [p for p in checkpoint_dir.glob("iter*.m") if p.is_file()]
    all_ckpts = sorted(all_ckpts, key=lambda p: parse_iter_idx(p))

    if not all_ckpts:
        raise FileNotFoundError(f"No iter*.m checkpoints found in {checkpoint_dir}")
    return all_ckpts


def evaluate_checkpoint(args, model, exp_enc, tokenizer, exp_tokenizer, test_loader, emotion_tests, z_test, device):
    model.eval()
    if exp_enc is not None:
        exp_enc.eval()

    y_pred = []
    y_test = []

    for batch_data in tqdm(test_loader, desc="Original", leave=False):
        with torch.no_grad():
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            targets = batch_data["labels"].to(device)

            if args.model_version in ["v4", "v5"]:
                if args.use_sliding_window:
                    batch_size_curr = targets.size(0)
                    out_bi_list = []
                    for i in range(batch_size_curr):
                        text = batch_data["news_text"][i]
                        explanation = batch_data["explanation"][i]
                        val_out = sliding_window_infer_v4(
                            model,
                            tokenizer,
                            text,
                            explanation,
                            args.window_size,
                            args.window_stride,
                            device,
                            max_windows=args.eval_max_windows,
                        )
                        out_bi_list.append(val_out)
                    val_out = torch.cat(out_bi_list, dim=0)
                else:
                    text_boundary = batch_data["text_boundary"]
                    _, val_out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        text_boundary=text_boundary[0] if len(text_boundary) > 0 else None,
                    )
            else:
                if exp_enc is not None:
                    input_ids_exp = batch_data["input_ids_exp"].to(device)
                    attention_mask_exp = batch_data["attention_mask_exp"].to(device)
                    exp_enc_out = exp_enc(input_ids=input_ids_exp, attention_mask=attention_mask_exp)
                    exp_feature = exp_enc_out.last_hidden_state.to(device)
                else:
                    exp_feature = None
                _, val_out = model(input_ids=input_ids, attention_mask=attention_mask, exp_feature=exp_feature,
                                   exp_attention_mask=attention_mask_exp)

            _, val_pred = val_out.max(dim=1)
            y_pred.append(val_pred)
            y_test.append(targets)

    y_pred = torch.cat(y_pred, dim=0)
    y_test = torch.cat(y_test, dim=0)
    y_true_np = y_test.detach().cpu().numpy()

    acc = accuracy_score(y_true_np, y_pred.detach().cpu().numpy())
    precision, recall, fscore, _ = score(y_true_np, y_pred.detach().cpu().numpy(), average="macro")

    emo_accs, emo_precs, emo_recs, emo_f1s = [], [], [], []
    emotion_results = {}

    for emo, x_test_res_e in emotion_tests.items():
        test_loader_res = create_eval_loader(args, x_test_res_e, y_true_np, z_test, tokenizer, exp_tokenizer)
        y_pred_res = []

        for batch_data in tqdm(test_loader_res, desc=f"Restyle-{emo}", leave=False):
            with torch.no_grad():
                input_ids_aug = batch_data["input_ids"].to(device)
                attention_mask_aug = batch_data["attention_mask"].to(device)

                if args.model_version in ["v4", "v5"]:
                    if args.use_sliding_window:
                        batch_size_curr = input_ids_aug.size(0)
                        out_bi_list = []
                        for i in range(batch_size_curr):
                            text_aug = batch_data["news_text"][i]
                            explanation_aug = batch_data["explanation"][i]
                            val_out_aug = sliding_window_infer_v4(
                                model,
                                tokenizer,
                                text_aug,
                                explanation_aug,
                                args.window_size,
                                args.window_stride,
                                device,
                                max_windows=args.eval_max_windows,
                            )
                            out_bi_list.append(val_out_aug)
                        val_out_aug = torch.cat(out_bi_list, dim=0)
                    else:
                        text_boundary_aug = batch_data["text_boundary"]
                        _, val_out_aug = model(
                            input_ids=input_ids_aug,
                            attention_mask=attention_mask_aug,
                            text_boundary=text_boundary_aug[0] if len(text_boundary_aug) > 0 else None,
                        )
                else:
                    if exp_enc is not None:
                        input_ids_exp = batch_data["input_ids_exp"].to(device)
                        attention_mask_exp = batch_data["attention_mask_exp"].to(device)
                        exp_enc_out = exp_enc(input_ids=input_ids_exp, attention_mask=attention_mask_exp)
                        exp_feature = exp_enc_out.last_hidden_state.to(device)
                    else:
                        exp_feature = None
                    _, val_out_aug = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug, exp_feature=exp_feature,
                                           exp_attention_mask=attention_mask_exp)

                _, val_pred_aug = val_out_aug.max(dim=1)
                y_pred_res.append(val_pred_aug)

        y_pred_res = torch.cat(y_pred_res, dim=0)
        acc_e = accuracy_score(y_true_np, y_pred_res.detach().cpu().numpy())
        precision_e, recall_e, fscore_e, _ = score(y_true_np, y_pred_res.detach().cpu().numpy(), average="macro")

        emo_accs.append(acc_e)
        emo_precs.append(precision_e)
        emo_recs.append(recall_e)
        emo_f1s.append(fscore_e)
        emotion_results[emo] = {
            "acc": acc_e,
            "precision": precision_e,
            "recall": recall_e,
            "f1": fscore_e,
        }

    acc_res = float(np.mean(emo_accs)) if emo_accs else 0.0
    precision_res = float(np.mean(emo_precs)) if emo_precs else 0.0
    recall_res = float(np.mean(emo_recs)) if emo_recs else 0.0
    fscore_res = float(np.mean(emo_f1s)) if emo_f1s else 0.0

    return {
        "orig": {"acc": acc, "precision": precision, "recall": recall, "f1": fscore},
        "restyle": {"acc": acc_res, "precision": precision_res, "recall": recall_res, "f1": fscore_res},
        "emotion_results": emotion_results,
    }


def summarize_metric(values):
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_backbones = {
        "roberta": "FacebookAI/roberta-base",
        "bert": "bert-base-uncased",
    }
    pretrained_backbone_name = pretrained_backbones[args.encoder_type]

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    selected_ckpts = collect_checkpoints(checkpoint_dir)

    backbone_cls, version_label = build_backbone(args)
    print(f"Using model version: {version_label} with backbone: {args.encoder_type}")
    print(f"Evaluating {len(selected_ckpts)} checkpoints from: {checkpoint_dir}")
    for ckpt in selected_ckpts:
        print(f"  - {ckpt.name}")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_backbone_name)
    exp_tokenizer = AutoTokenizer.from_pretrained(pretrained_backbone_name)

    _, x_test, _, _, y_test, _, z_test = load_articles(args.dataset_name)
    test_loader = create_eval_loader(args, x_test, y_test, z_test, tokenizer, exp_tokenizer)

    emotion_list = [e.strip() for e in args.test_emotions.split(",") if e.strip()]
    emotion_tests = load_emotion_tests(args.dataset_name, emotion_list)

    if not emotion_tests:
        print("[WARN] No emotion test sets loaded. Restyle metrics will be 0.")

    per_round = []
    all_emotion_results = []

    for idx, ckpt_path in enumerate(selected_ckpts):
        print("\n" + "=" * 80)
        print(f"Round {idx + 1}/{len(selected_ckpts)} - {ckpt_path.name}")
        print("=" * 80)

        model = TransformerClassifier(
            args=args,
            n_classes=4,
            backbone_cls=backbone_cls,
            pretrained_name=pretrained_backbone_name,
        ).float().to(device)

        checkpoint = torch.load(ckpt_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
            exp_encoder_state_dict = checkpoint.get("exp_encoder_state_dict")
        else:
            model_state_dict = checkpoint
            exp_encoder_state_dict = None
        model.load_state_dict(model_state_dict)

        if args.model_version not in ["v4", "v5"]:
            if args.encoder_type == "roberta":
                exp_enc = RobertaEncoderResults.from_pretrained(pretrained_backbone_name, return_dict=True).to(device)
            else:
                exp_enc = HFBertModel.from_pretrained(pretrained_backbone_name, return_dict=True).to(device)
            if exp_encoder_state_dict is not None:
                exp_enc.load_state_dict(exp_encoder_state_dict)
            else:
                print("[WARN] Legacy checkpoint has no trained explanation encoder; results are not strictly reproducible.")
        else:
            exp_enc = None

        result = evaluate_checkpoint(
            args,
            model,
            exp_enc,
            tokenizer,
            exp_tokenizer,
            test_loader,
            emotion_tests,
            z_test,
            device,
        )

        per_round.append(result)
        all_emotion_results.append(result["emotion_results"])

        print("Original:")
        print(
            f"  Acc={result['orig']['acc']:.4f} | "
            f"Prec={result['orig']['precision']:.4f} | "
            f"Rec={result['orig']['recall']:.4f} | "
            f"F1={result['orig']['f1']:.4f}"
        )
        print("Restyle:")
        print(
            f"  Acc={result['restyle']['acc']:.4f} | "
            f"Prec={result['restyle']['precision']:.4f} | "
            f"Rec={result['restyle']['recall']:.4f} | "
            f"F1={result['restyle']['f1']:.4f}"
        )

    orig_accs = [r["orig"]["acc"] for r in per_round]
    orig_precs = [r["orig"]["precision"] for r in per_round]
    orig_recs = [r["orig"]["recall"] for r in per_round]
    orig_f1s = [r["orig"]["f1"] for r in per_round]

    res_accs = [r["restyle"]["acc"] for r in per_round]
    res_precs = [r["restyle"]["precision"] for r in per_round]
    res_recs = [r["restyle"]["recall"] for r in per_round]
    res_f1s = [r["restyle"]["f1"] for r in per_round]

    print("\n" + "=" * 80)
    print(f"FINAL AVERAGE OVER {len(per_round)} ROUNDS")
    print("=" * 80)

    m_acc, s_acc = summarize_metric(orig_accs)
    m_prec, s_prec = summarize_metric(orig_precs)
    m_rec, s_rec = summarize_metric(orig_recs)
    m_f1, s_f1 = summarize_metric(orig_f1s)
    print("Original:")
    print(f"  Accuracy:  {m_acc:.4f} +- {s_acc:.4f}")
    print(f"  Precision: {m_prec:.4f} +- {s_prec:.4f}")
    print(f"  Recall:    {m_rec:.4f} +- {s_rec:.4f}")
    print(f"  F1:        {m_f1:.4f} +- {s_f1:.4f}")

    rm_acc, rs_acc = summarize_metric(res_accs)
    rm_prec, rs_prec = summarize_metric(res_precs)
    rm_rec, rs_rec = summarize_metric(res_recs)
    rm_f1, rs_f1 = summarize_metric(res_f1s)
    print("Restyle:")
    print(f"  Accuracy:  {rm_acc:.4f} +- {rs_acc:.4f}")
    print(f"  Precision: {rm_prec:.4f} +- {rs_prec:.4f}")
    print(f"  Recall:    {rm_rec:.4f} +- {rs_rec:.4f}")
    print(f"  F1:        {rm_f1:.4f} +- {rs_f1:.4f}")

    emotion_names = sorted({emo for round_result in all_emotion_results for emo in round_result.keys()})
    if emotion_names:
        print("\nPer-emotion Restyle (avg +- std):")
        for emo in emotion_names:
            emo_accs = [round_result[emo]["acc"] for round_result in all_emotion_results if emo in round_result]
            emo_precs = [round_result[emo]["precision"] for round_result in all_emotion_results if emo in round_result]
            emo_recs = [round_result[emo]["recall"] for round_result in all_emotion_results if emo in round_result]
            emo_f1s = [round_result[emo]["f1"] for round_result in all_emotion_results if emo in round_result]

            ea, esa = summarize_metric(emo_accs)
            ep, esp = summarize_metric(emo_precs)
            er, esr = summarize_metric(emo_recs)
            ef, esf = summarize_metric(emo_f1s)
            print(f"  {emo}:")
            print(f"    Acc={ea:.4f} +- {esa:.4f} | Prec={ep:.4f} +- {esp:.4f} | Rec={er:.4f} +- {esr:.4f} | F1={ef:.4f} +- {esf:.4f}")


if __name__ == "__main__":
    main()
