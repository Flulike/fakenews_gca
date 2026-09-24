import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import argparse
import numpy as np
import sys, os
import importlib
from datetime import datetime
from pathlib import Path
sys.path.append(os.getcwd())
from utils.load_data import *
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from roberta_encoder import RobertaEncoderResults
from transformers import BertModel as HFBertModel

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='politifact', type=str)
parser.add_argument('--model_name', default='Pretrained-LM', type=str)
parser.add_argument('--iters', default=2, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--n_epochs', default=5, type=int)
parser.add_argument('--fusion_layer', default=11, type=int)
parser.add_argument('--dim_common', default=256, type=int)
parser.add_argument('--n_attn_heads', default=1, type=int)
parser.add_argument('--model_version', default='v1', type=str, choices=['v1', 'v2', 'v3', 'v4', 'v5'], 
                    help='Choose model version: v1 (multi-head attention), v2 (token-wise attention), v3 (gated+mamba), v4 (concatenation+gate+decoder), or v5 (enhanced method 2 with improved loss)')
parser.add_argument('--encoder_type', default='roberta', type=str, choices=['roberta', 'bert'],
                    help='Select pretrained backbone family (RoBERTa or BERT)')
parser.add_argument('--test_emotions', default='', type=str,
                    help='Comma-separated emotions for adversarial testing; empty means auto-detect all available')
parser.add_argument('--use_sliding_window', action='store_true',
                    help='Enable sliding window for v4 to handle long concatenated sequences')
parser.add_argument('--window_size', default=512, type=int,
                    help='Sliding window size (tokens) for v4')
parser.add_argument('--window_stride', default=256, type=int,
                    help='Sliding window stride (tokens) for v4')
parser.add_argument('--train_windows_per_sample', default=1, type=int,
                    help='Number of windows per sample to use during training when sliding window is enabled')
parser.add_argument('--eval_max_windows', default=32, type=int,
                    help='Maximum number of windows per sample to use during evaluation (to cap memory/time)')
parser.add_argument('--use_match_loss', action='store_true',
                    help='Enable InfoNCE match loss between text and explanation features')
parser.add_argument('--disable_gate', action='store_true',
                    help='Disable gate mechanism in fusion layer (for ablation study)')
parser.add_argument('--distorted', action='store_true',
                    help='Shift training explanations by one position to break text-explanation alignment')
parser.add_argument('--run_name', default='', type=str,
                    help='Run identifier used for logs/checkpoints; if empty, auto-generated')

args = parser.parse_args()


def build_run_name(datasetname: str) -> str:
    """Build a stable run name shared by logs and checkpoint directory."""
    if args.run_name:
        return args.run_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [timestamp, datasetname, args.model_version]
    if args.disable_gate:
        name_parts.append("nogate")
    if args.use_match_loss:
        name_parts.append("matchloss")
    if args.encoder_type != 'roberta':
        name_parts.append(args.encoder_type)
    return "_".join(name_parts)


def build_checkpoint_path(datasetname: str, run_name: str, iter_idx: int) -> Path:
    """Save checkpoints into checkpoints/<dataset>/<run_name>/iterX.m."""
    checkpoint_dir = resolve_checkpoint_run_dir(datasetname, run_name)
    return checkpoint_dir / f"iter{iter_idx}.m"


def resolve_checkpoint_run_dir(datasetname: str, run_name: str) -> Path:
    """Return the checkpoint directory for this run, preferring the exact run name."""
    base_dir = Path("checkpoints") / datasetname
    exact_dir = base_dir / run_name

    if exact_dir.exists():
        return exact_dir

    if base_dir.exists():
        candidate_dirs = [
            path for path in base_dir.iterdir()
            if path.is_dir() and run_name in path.name
        ]
        if candidate_dirs:
            return max(candidate_dirs, key=lambda path: path.stat().st_mtime)

    exact_dir.mkdir(parents=True, exist_ok=True)
    return exact_dir


def distort_explanations(explanations):
    """Rotate explanations by one position for training ablations."""
    if not explanations:
        return explanations
    if len(explanations) == 1:
        return explanations
    return explanations[1:] + explanations[:1]

# 根据命令行参数选择模型版本和底层编码器
MODEL_REGISTRY = {
    'v1': ('modeling_roberta', 'v1 (multi-head attention)') ,
    'v2': ('modeling_roberta2', 'v2 (token-wise attention)') ,
    'v3': ('modeling_roberta3', 'v3 (gated+mamba)') ,
    'v4': ('modeling_roberta4', 'v4 (concatenation+gate+decoder)') ,
    'v5': ('modeling_roberta5', 'v5 (enhanced method 2 with improved loss)') ,
}

if args.model_version not in MODEL_REGISTRY:
    raise ValueError(f"Unknown model version: {args.model_version}")

module_name, version_label = MODEL_REGISTRY[args.model_version]
model_module = importlib.import_module(module_name)

backbone_options = {'roberta': getattr(model_module, 'RobertaModel')}
if hasattr(model_module, 'BertModel'):
    backbone_options['bert'] = getattr(model_module, 'BertModel')

if args.encoder_type not in backbone_options:
    raise ValueError(f"Encoder type '{args.encoder_type}' is not supported for model version '{args.model_version}'.")

BackboneModel = backbone_options[args.encoder_type]
print(f"Using model version: {version_label} with backbone: {args.encoder_type}")

PRETRAINED_BACKBONES = {
    'roberta': 'FacebookAI/roberta-base',
    'bert': 'bert-base-uncased',
}

pretrained_backbone_name = PRETRAINED_BACKBONES[args.encoder_type]

device = torch.device("cuda")
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(0)
        

class NewsDatasetAug(Dataset):
    def __init__(self, texts, aug_texts1, aug_texts2, labels, explanation, fg_label, aug_fg1, aug_fg2, tokenizer, exp_tokenizer, max_len):
        self.texts = texts
        self.aug_texts1 = aug_texts1
        self.aug_texts2 = aug_texts2
        self.explanation = explanation
        self.tokenizer = tokenizer
        self.exp_tokenizer = exp_tokenizer
        self.max_len = max_len
        self.labels = labels
        self.fg_label = fg_label
        self.aug_fg1 = aug_fg1
        self.aug_fg2 = aug_fg2

    def __getitem__(self, item):
        text = self.texts[item]
        aug_text1 = self.aug_texts1[item]
        aug_text2 = self.aug_texts2[item]
        explanation=self.explanation[item]
        label = self.labels[item]
        fg_label = self.fg_label[item]
        aug_fg1 = self.aug_fg1[item]
        aug_fg2 = self.aug_fg2[item]
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        aug1_encoding = self.tokenizer.encode_plus(aug_text1, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        aug2_encoding = self.tokenizer.encode_plus(aug_text2, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')

        exp_encoding = self.exp_tokenizer.encode_plus(explanation, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'input_ids_aug1': aug1_encoding['input_ids'].flatten(),
            'input_ids_aug2': aug2_encoding['input_ids'].flatten(),
            'input_ids_exp': exp_encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attention_mask_aug1': aug1_encoding['attention_mask'].flatten(),
            'attention_mask_aug2': aug2_encoding['attention_mask'].flatten(),
            'attention_mask_exp': exp_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg_label),
            'fg_label_aug1': torch.FloatTensor(aug_fg1),
            'fg_label_aug2': torch.FloatTensor(aug_fg2),
        }

    def __len__(self):
        return len(self.texts)

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
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')
        exp_encoding = self.exp_tokenizer.encode_plus(explanation, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='pt')
        return {
            'news_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'explanation': explanation,
            'input_ids_exp': exp_encoding['input_ids'].flatten(),
            'attention_mask_exp': exp_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


class NewsDatasetConcatenated(Dataset):
    """Dataset for v4 model that concatenates text and explanation"""
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
        
        # 简单方案：文本和解释各占256个token，多的就截断，然后拼接
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=256)
        exp_tokens = self.tokenizer.encode(explanation, add_special_tokens=False, truncation=True, max_length=256)
        
        # 确保总长度不超过512
        if len(text_tokens) + len(exp_tokens) > 509:  # 512 - 3 = 509 (为特殊token留空间)
            # 平均分配
            text_tokens = text_tokens[:254]
            exp_tokens = exp_tokens[:254]
        
        # 直接拼接：[CLS] text [SEP] explanation [SEP]
        input_ids = [self.tokenizer.cls_token_id] + text_tokens + [self.tokenizer.sep_token_id] + exp_tokens + [self.tokenizer.sep_token_id]
        
        # 记录文本边界位置，用于gate机制
        text_boundary = len(text_tokens) + 2  # +2 for CLS and first SEP
        
        # Pad到max_len (512)
        attention_mask = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
        
        return {
            'news_text': text,
            'explanation': explanation,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'text_boundary': text_boundary,  # For gate mechanism
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


class NewsDatasetAugConcatenated(Dataset):
    """Augmented dataset for v4 model that concatenates text and explanation"""
    def __init__(self, texts, aug_texts1, aug_texts2, labels, explanation, fg_label, aug_fg1, aug_fg2, tokenizer, max_len):
        self.texts = texts
        self.aug_texts1 = aug_texts1
        self.aug_texts2 = aug_texts2
        self.explanation = explanation
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels
        self.fg_label = fg_label
        self.aug_fg1 = aug_fg1
        self.aug_fg2 = aug_fg2

    def _concatenate_text_explanation(self, text, explanation):
        """Helper method to concatenate text and explanation with proper tokenization"""
        # 简单方案：文本和解释各占256个token，多的就截断，然后拼接
        text_tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=256)
        exp_tokens = self.tokenizer.encode(explanation, add_special_tokens=False, truncation=True, max_length=256)
        
        # 确保总长度不超过512
        if len(text_tokens) + len(exp_tokens) > 509:  # 512 - 3 = 509 (为特殊token留空间)
            # 平均分配
            text_tokens = text_tokens[:254]
            exp_tokens = exp_tokens[:254]
        
        # 直接拼接：[CLS] text [SEP] explanation [SEP]
        input_ids = [self.tokenizer.cls_token_id] + text_tokens + [self.tokenizer.sep_token_id] + exp_tokens + [self.tokenizer.sep_token_id]
        
        # 记录文本边界位置，用于gate机制
        text_boundary = len(text_tokens) + 2  # +2 for CLS and first SEP
        
        # Pad到max_len (512)
        attention_mask = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'text_boundary': text_boundary
        }

    def __getitem__(self, item):
        text = self.texts[item]
        aug_text1 = self.aug_texts1[item]
        aug_text2 = self.aug_texts2[item]
        explanation = self.explanation[item]
        label = self.labels[item]
        fg_label = self.fg_label[item]
        aug_fg1 = self.aug_fg1[item]
        aug_fg2 = self.aug_fg2[item]
        
        # Create concatenated encodings for all three text versions
        encoding = self._concatenate_text_explanation(text, explanation)
        aug1_encoding = self._concatenate_text_explanation(aug_text1, explanation)
        aug2_encoding = self._concatenate_text_explanation(aug_text2, explanation)
        
        return {
            'news_text': text,
            'explanation': explanation,
            'news_text_aug1': aug_text1,
            'news_text_aug2': aug_text2,
            'input_ids': encoding['input_ids'],
            'input_ids_aug1': aug1_encoding['input_ids'],
            'input_ids_aug2': aug2_encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'attention_mask_aug1': aug1_encoding['attention_mask'],
            'attention_mask_aug2': aug2_encoding['attention_mask'],
            'text_boundary': encoding['text_boundary'],
            'text_boundary_aug1': aug1_encoding['text_boundary'],
            'text_boundary_aug2': aug2_encoding['text_boundary'],
            'labels': torch.tensor(label, dtype=torch.long),
            'fg_label': torch.FloatTensor(fg_label),
            'fg_label_aug1': torch.FloatTensor(aug_fg1),
            'fg_label_aug2': torch.FloatTensor(aug_fg2),
        }

    def __len__(self):
        return len(self.texts)


class TransformerClassifier(nn.Module):
    def __init__(self, n_classes, backbone_cls, encoder_type, pretrained_name):
        super().__init__()
        self.model_version = args.model_version
        self.encoder_type = encoder_type
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
        if self.model_version == 'v4':
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, text_boundary=text_boundary)
        elif self.model_version == 'v2':
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                exp_feature=exp_feature,
                exp_attention_mask=exp_attention_mask,
            )
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, exp_feature=exp_feature)

        pooled = getattr(outputs, 'pooler_output', None)
        if pooled is None and isinstance(outputs, (tuple, list)):
            pooled = outputs[1]
        pooled_outputs = self.dropout(pooled)
        output = self.fc_out(pooled_outputs)
        binary_output = self.binary_transform(pooled_outputs)

        gate_values = getattr(outputs, 'gate', None)
        if args.use_match_loss and gate_values is not None:
            return output, binary_output, gate_values
        return output, binary_output


def build_concat_ids(tokenizer, text, explanation, max_len=512):
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    exp_tokens = tokenizer.encode(explanation, add_special_tokens=False)
    # reserve 3 specials
    max_total = max_len - 3
    if len(text_tokens) + len(exp_tokens) > max_total:
        # proportional truncation
        t_keep = max(1, int(max_total * len(text_tokens) / (len(text_tokens) + len(exp_tokens))))
        e_keep = max_total - t_keep
        text_tokens = text_tokens[:t_keep]
        exp_tokens = exp_tokens[:e_keep]
    input_ids = [tokenizer.cls_token_id] + text_tokens + [tokenizer.sep_token_id] + exp_tokens + [tokenizer.sep_token_id]
    text_boundary = len(text_tokens) + 2
    attention_mask = [1] * len(input_ids)
    # pad
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        text_boundary,
    )


def sliding_window_infer_v4(model, tokenizer, text, explanation, window_size=512, stride=256, device='cuda', require_grad=False, max_windows=None, train_windows_per_sample=None):
    # build full tokens without pre-truncation
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    exp_tokens = tokenizer.encode(explanation, add_special_tokens=False)
    # compose full concatenated sequence with boundary (no padding here)
    full_ids = [tokenizer.cls_token_id] + text_tokens + [tokenizer.sep_token_id] + exp_tokens + [tokenizer.sep_token_id]
    boundary = len(text_tokens) + 2
    total_len = len(full_ids)
    if total_len <= window_size:
        ids, mask, tb = build_concat_ids(tokenizer, text, explanation, max_len=window_size)
        ids = ids.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        out, out_bi = model(input_ids=ids, attention_mask=mask, text_boundary=tb)
        return out, out_bi
    # windows
    logits_list = []
    logits_bi_list = []
    starts = list(range(0, max(1, total_len - window_size + 1), stride))
    # cap number of windows for eval
    if max_windows is not None and len(starts) > max_windows:
        # select evenly spaced indices
        idxs = np.linspace(0, len(starts) - 1, num=max_windows, dtype=int)
        starts = [starts[i] for i in idxs]
    # limit number of gradient windows for training
    grad_budget = train_windows_per_sample if (require_grad and train_windows_per_sample is not None) else None
    for start in starts:
        end = min(start + window_size, total_len)
        window_ids = full_ids[start:end]
        # compute window boundary relative to window
        tb_win = boundary - start
        tb_win = int(max(0, min(tb_win, len(window_ids))))
        # pad
        pad_len = window_size - len(window_ids)
        input_ids = window_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * len(window_ids) + [0] * pad_len
        ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
        if require_grad and (grad_budget is None or grad_budget > 0):
            out, out_bi = model(input_ids=ids, attention_mask=mask, text_boundary=tb_win)
            if grad_budget is not None:
                grad_budget -= 1
        else:
            with torch.no_grad():
                out, out_bi = model(input_ids=ids, attention_mask=mask, text_boundary=tb_win)
        logits_list.append(out)
        logits_bi_list.append(out_bi)
    # aggregate (mean)
    logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
    logits_bi = torch.mean(torch.stack(logits_bi_list, dim=0), dim=0)
    return logits, logits_bi

def create_train_loader(contents, contents_aug1, contents_aug2, labels, explanation, fg_label, aug_fg1, aug_fg2, tokenizer, exp_tokenizer, max_len, batch_size):
    if args.model_version in ['v4', 'v5']:
        # For v4, v5: use concatenated dataset (no separate exp_tokenizer needed)
        ds = NewsDatasetAugConcatenated(texts=contents, aug_texts1=contents_aug1, aug_texts2=contents_aug2, 
                                      labels=np.array(labels), explanation=explanation, 
                                      fg_label=fg_label, aug_fg1=aug_fg1, aug_fg2=aug_fg2, 
                                      tokenizer=tokenizer, max_len=max_len)
    else:
        # For v1, v2, v3: use original dataset
        ds = NewsDatasetAug(texts=contents, aug_texts1=contents_aug1, aug_texts2=contents_aug2, labels=np.array(labels), explanation=explanation, \
                            fg_label=fg_label, aug_fg1=aug_fg1, aug_fg2=aug_fg2, tokenizer=tokenizer, exp_tokenizer=exp_tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

def create_eval_loader(contents, labels, explanation, tokenizer, exp_tokenizer, max_len, batch_size):
    if args.model_version in ['v4', 'v5']:
        # For v4, v5: use concatenated dataset (no separate exp_tokenizer needed)
        ds = NewsDatasetConcatenated(texts=contents, labels=np.array(labels), explanation=explanation, 
                                   tokenizer=tokenizer, max_len=max_len)
    else:
        # For v1, v2, v3: use original dataset
        ds = NewsDataset(texts=contents, labels=np.array(labels), explanation=explanation, tokenizer=tokenizer, exp_tokenizer=exp_tokenizer, max_len=max_len)
    
    return DataLoader(ds, batch_size=batch_size, num_workers=0)



def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def info_nce_loss(msg_embeds, exp_embeds, temperature=0.1):
    """
    InfoNCE loss for contrastive learning between message and explanation embeddings
    """
    msg_embeds = msg_embeds.mean(dim=1)
    exp_embeds = exp_embeds.mean(dim=1)
    msg_embeds = F.normalize(msg_embeds, dim=1)
    exp_embeds = F.normalize(exp_embeds, dim=1)
    logits = torch.matmul(msg_embeds, exp_embeds.T)
    labels = torch.arange(msg_embeds.size(0)).to(msg_embeds.device)
    logits /= temperature
    return F.cross_entropy(logits, labels)

def bi_info_nce_loss(msg_embeds, exp_embeds, temperature=0.1):
    """
    双向 InfoNCE loss
    msg_embeds: [batch, seq, dim]
    exp_embeds: [batch, seq, dim]
    """
    # 池化
    msg_embeds = msg_embeds.mean(dim=1)
    exp_embeds = exp_embeds.mean(dim=1)
    # 归一化
    msg_embeds = F.normalize(msg_embeds, dim=1)
    exp_embeds = F.normalize(exp_embeds, dim=1)
    # 相似度矩阵
    logits = torch.matmul(msg_embeds, exp_embeds.T) / temperature
    labels = torch.arange(msg_embeds.size(0)).to(msg_embeds.device)
    # 消息→解释
    loss_msg2exp = F.cross_entropy(logits, labels)
    # 解释→消息
    loss_exp2msg = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_msg2exp + loss_exp2msg)


def train_model(tokenizer, exp_tokenizer, max_len, n_epochs, batch_size, datasetname, iter, run_name):

    x_train, x_test, x_test_res, y_train, y_test, z_train, z_test = load_articles(datasetname)
    print(f"Loaded data - Train: {len(x_train)}, Test: {len(x_test)}, Explanations: {len(z_train) if z_train else 'None'}")
    print(f"Label distribution - Train: {np.unique(y_train, return_counts=True)}")
    if args.distorted:
        z_train = distort_explanations(z_train)
        print("Training explanations are distorted by one position.")
    print(f"Sample explanation: {z_train[0] if z_train and len(z_train) > 0 else 'No explanations'}")
    test_loader = create_eval_loader(x_test, y_test, z_test, tokenizer, exp_tokenizer, max_len, batch_size)
    # multi-emotion adversarial test sets
    emotion_list = [e.strip() for e in args.test_emotions.split(',') if e.strip()]
    emotion_tests = load_emotion_tests(args.dataset_name, emotion_list)

    model = TransformerClassifier(
        n_classes=4,
        backbone_cls=BackboneModel,
        encoder_type=args.encoder_type,
        pretrained_name=pretrained_backbone_name,
    ).float().to(device)
    with torch.no_grad():
        # 版本1、版本2和版本3都有_multi_head_attn_1属性
        if args.model_version in ['v1', 'v2', 'v3']:
            model.backbone.encoder._multi_head_attn_1._reset_parameters()
            
            # 版本3需要额外的初始化
            if args.model_version == 'v3':
                # 初始化gated mamba fusion的参数
                if hasattr(model.backbone.encoder, 'gated_mamba_fusion') and model.backbone.encoder.gated_mamba_fusion is not None:
                    # 重新初始化Mamba组件的参数
                    for module in model.backbone.encoder.gated_mamba_fusion.modules():
                        if isinstance(module, torch.nn.Linear):
                            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                            if module.bias is not None:
                                torch.nn.init.zeros_(module.bias)
                        elif isinstance(module, torch.nn.LayerNorm):
                            torch.nn.init.ones_(module.weight)
                            torch.nn.init.zeros_(module.bias)
                
                # 初始化gate layer
                for module in model.backbone.encoder.gate_layer:
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                            
                # 初始化exp_gate
                for module in model.backbone.encoder.exp_gate:
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                
                # 初始化beta参数
                model.backbone.encoder.beta.data.fill_(0.1)
    #model.backbone.encoder._multi_head_attn_1.in_proj_bias.data.zero_()
    # For v4 and v5, we don't need a separate explanation encoder
    if args.model_version not in ['v4', 'v5']:
        if args.encoder_type == 'roberta':
            exp_enc = RobertaEncoderResults.from_pretrained(pretrained_backbone_name, return_dict=True).to(device)
        elif args.encoder_type == 'bert':
            exp_enc = HFBertModel.from_pretrained(pretrained_backbone_name, return_dict=True).to(device)
        else:
            exp_enc = None
    else:
        exp_enc = None
        
    train_losses = []
    train_accs = []
    # 优化器需要包含explanation encoder的参数
    # 版本3和v5更复杂，使用更小的学习率和权重衰减
    if args.model_version in ['v3', 'v5']:
        lr = 5e-6 if args.model_version == 'v3' else 1e-5  # v5 uses slightly higher lr than v3
        weight_decay = 0.01  # 添加权重衰减
    else:
        lr = 2e-5
        weight_decay = 0.0
    
    if exp_enc is not None:
        optimizer = AdamW(list(model.parameters()) + list(exp_enc.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(n_epochs):
        model.train()
        x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t = load_reframing(args.dataset_name)
        if exp_enc is not None:
            exp_enc.train()
        print(f"Fine-grain labels shape: {np.array(y_train_fg).shape if y_train_fg is not None else 'None'}")
        print(f"Fine-grain labels sample: {y_train_fg[:3] if y_train_fg is not None else 'None'}")
        train_loader = create_train_loader(x_train, x_train_res1, x_train_res2, y_train, z_train, y_train_fg, y_train_fg_m, y_train_fg_t, tokenizer, exp_tokenizer, max_len, batch_size)

        avg_loss = []
        avg_acc = []
        batch_idx = 0

        for Batch_data in tqdm(train_loader):
            input_ids = Batch_data["input_ids"].to(device)
            attention_mask = Batch_data["attention_mask"].to(device)
            input_ids_aug1 = Batch_data["input_ids_aug1"].to(device)
            attention_mask_aug1 = Batch_data["attention_mask_aug1"].to(device)
            input_ids_aug2 = Batch_data["input_ids_aug2"].to(device)
            attention_mask_aug2 = Batch_data["attention_mask_aug2"].to(device)
            targets = Batch_data["labels"].to(device)
            fg_labels = Batch_data["fg_label"].to(device)
            fg_labels_aug1 = Batch_data["fg_label_aug1"].to(device)
            fg_labels_aug2 = Batch_data["fg_label_aug2"].to(device)
            
            if args.model_version in ['v4', 'v5']:
                if args.use_sliding_window:
                    batch_size_curr = targets.size(0)
                    out_list, out_bi_list = [], []
                    out1_list, out1_bi_list = [], []
                    out2_list, out2_bi_list = [], []
                    for i in range(batch_size_curr):
                        text = Batch_data["news_text"][i]
                        text1 = Batch_data["news_text_aug1"][i]
                        text2 = Batch_data["news_text_aug2"][i]
                        explanation = Batch_data["explanation"][i]
                        o, obi = sliding_window_infer_v4(model, tokenizer, text, explanation, args.window_size, args.window_stride, device, require_grad=True, train_windows_per_sample=args.train_windows_per_sample)
                        o1, obi1 = sliding_window_infer_v4(model, tokenizer, text1, explanation, args.window_size, args.window_stride, device, require_grad=True, train_windows_per_sample=args.train_windows_per_sample)
                        o2, obi2 = sliding_window_infer_v4(model, tokenizer, text2, explanation, args.window_size, args.window_stride, device, require_grad=True, train_windows_per_sample=args.train_windows_per_sample)
                        out_list.append(o)
                        out_bi_list.append(obi)
                        out1_list.append(o1)
                        out1_bi_list.append(obi1)
                        out2_list.append(o2)
                        out2_bi_list.append(obi2)
                    out_labels = torch.cat(out_list, dim=0)
                    out_labels_bi = torch.cat(out_bi_list, dim=0)
                    out_labels_aug1 = torch.cat(out1_list, dim=0)
                    out_labels_bi_aug1 = torch.cat(out1_bi_list, dim=0)
                    out_labels_aug2 = torch.cat(out2_list, dim=0)
                    out_labels_bi_aug2 = torch.cat(out2_bi_list, dim=0)
                else:
                    # For v4: use concatenated inputs with text_boundary (no token_type_ids needed)
                    text_boundary = Batch_data["text_boundary"]
                    text_boundary_aug1 = Batch_data["text_boundary_aug1"]
                    text_boundary_aug2 = Batch_data["text_boundary_aug2"]
                    
                    # 添加调试信息
                    if batch_idx == 0 and epoch == 0:
                        print(f"V4 Input shape: {input_ids.shape}")
                        print(f"Text boundary: {text_boundary[0] if len(text_boundary) > 0 else 'None'}")
                        print(f"Targets shape: {targets.shape}, unique values: {torch.unique(targets)}")
                        print(f"FG labels shape: {fg_labels.shape}, range: [{fg_labels.min():.3f}, {fg_labels.max():.3f}]")
                        print(f"Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")
                        print(f"Vocab size: {tokenizer.vocab_size}")
                    
                    # Model forward passes for v4 (不使用token_type_ids)
                    model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                                        text_boundary=text_boundary[0] if len(text_boundary) > 0 else None)
                    if args.use_match_loss and len(model_outputs) == 3:
                        out_labels, out_labels_bi, gate_1 = model_outputs
                    else:
                        out_labels, out_labels_bi = model_outputs
                        gate_1 = None
                    
                    model_outputs_aug1 = model(input_ids=input_ids_aug1, attention_mask=attention_mask_aug1, 
                                             text_boundary=text_boundary_aug1[0] if len(text_boundary_aug1) > 0 else None)
                    if args.use_match_loss and len(model_outputs_aug1) == 3:
                        out_labels_aug1, out_labels_bi_aug1, gate_2 = model_outputs_aug1
                    else:
                        out_labels_aug1, out_labels_bi_aug1 = model_outputs_aug1
                        gate_2 = None
                    
                    model_outputs_aug2 = model(input_ids=input_ids_aug2, attention_mask=attention_mask_aug2, 
                                             text_boundary=text_boundary_aug2[0] if len(text_boundary_aug2) > 0 else None)
                    if args.use_match_loss and len(model_outputs_aug2) == 3:
                        out_labels_aug2, out_labels_bi_aug2, gate_3 = model_outputs_aug2
                    else:
                        out_labels_aug2, out_labels_bi_aug2 = model_outputs_aug2
                        gate_3 = None
            else:
                # For v1, v2, v3: use exp_feature approach when available
                if exp_enc is not None:
                    input_ids_exp = Batch_data["input_ids_exp"].to(device)
                    attention_mask_exp = Batch_data["attention_mask_exp"].to(device)
                    
                    # Debug: check explanation input_ids range
                    if batch_idx == 0 and epoch == 0:
                        print(f"Explanation Input IDs - min: {input_ids_exp.min().item()}, max: {input_ids_exp.max().item()}")
                        print(f"Exp encoder vocab size: {exp_enc.config.vocab_size}")
                    
                    exp_enc_out = exp_enc(input_ids=input_ids_exp, attention_mask=attention_mask_exp)
                    exp_feature = exp_enc_out.last_hidden_state.to(device)

                    if batch_idx == 0 and epoch == 0:
                        print(f"Explanation feature shape: {exp_feature.shape}")
                        print(f"Targets shape: {targets.shape}, unique values: {torch.unique(targets)}")
                        print(f"FG labels shape: {fg_labels.shape}, range: [{fg_labels.min():.3f}, {fg_labels.max():.3f}]")
                else:
                    exp_feature = None

                # Debug: check input_ids and position info (only first batch of first epoch)
                if batch_idx == 0 and epoch == 0:
                    print(f"Input IDs - min: {input_ids.min().item()}, max: {input_ids.max().item()}")
                    print(f"Model vocab size: {model.backbone.config.vocab_size}")
                    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
                    print(f"Max position embeddings: {model.backbone.config.max_position_embeddings}")
                    print(f"Padding idx: {model.backbone.embeddings.padding_idx}")
                
                # Model forward passes for v1, v2, v3
                model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, exp_feature=exp_feature,
                                      exp_attention_mask=attention_mask_exp)
                if args.use_match_loss and len(model_outputs) == 3:
                    out_labels, out_labels_bi, gate_1 = model_outputs
                else:
                    out_labels, out_labels_bi = model_outputs
                    gate_1 = None

                model_outputs_aug1 = model(input_ids=input_ids_aug1, attention_mask=attention_mask_aug1, exp_feature=exp_feature,
                                           exp_attention_mask=attention_mask_exp)
                if args.use_match_loss and len(model_outputs_aug1) == 3:
                    out_labels_aug1, out_labels_bi_aug1, gate_2 = model_outputs_aug1
                else:
                    out_labels_aug1, out_labels_bi_aug1 = model_outputs_aug1
                    gate_2 = None

                model_outputs_aug2 = model(input_ids=input_ids_aug2, attention_mask=attention_mask_aug2, exp_feature=exp_feature,
                                           exp_attention_mask=attention_mask_exp)
                if args.use_match_loss and len(model_outputs_aug2) == 3:
                    out_labels_aug2, out_labels_bi_aug2, gate_3 = model_outputs_aug2
                else:
                    out_labels_aug2, out_labels_bi_aug2 = model_outputs_aug2
                    gate_3 = None
            
            # 使用BCEWithLogitsLoss，它内部会应用sigmoid，更稳定
            fg_criterion = nn.BCEWithLogitsLoss()
            # 确保fg_labels在[0,1]范围内
            fg_labels = torch.clamp(fg_labels, 0, 1)
            fg_labels_aug1 = torch.clamp(fg_labels_aug1, 0, 1)
            fg_labels_aug2 = torch.clamp(fg_labels_aug2, 0, 1)
            # BCEWithLogitsLoss不需要手动应用sigmoid
            finegrain_loss = (fg_criterion(out_labels, fg_labels) + fg_criterion(out_labels_aug1, fg_labels_aug1) + \
                               fg_criterion(out_labels_aug2, fg_labels_aug2)) / 3

            out_probs = F.softmax(out_labels_bi, dim = -1)
            aug_log_prob1 = F.log_softmax(out_labels_bi_aug1, dim = -1)
            aug_log_prob2 = F.log_softmax(out_labels_bi_aug2, dim = -1)
            #以上需改
            sup_criterion = nn.CrossEntropyLoss()
            sup_loss = sup_criterion(out_labels_bi, targets)

            cons_criterion = nn.KLDivLoss(reduction = 'batchmean')
            cons_loss = 0.5 * cons_criterion(aug_log_prob1, out_probs) + 0.5 * cons_criterion(aug_log_prob2, out_probs)
            
            # Add InfoNCE match loss if enabled
            match_loss = 0.0
            if args.use_match_loss and args.model_version not in ['v4', 'v5']:
                # Only for v1, v2, v3 models that have gate values and separate explanation encoder
                if gate_1 is not None and gate_2 is not None and gate_3 is not None and exp_enc is not None:
                    # Get text embeddings from explanation encoder for match loss calculation
                    input_enc_out_bk = exp_enc(input_ids=input_ids, attention_mask=attention_mask)
                    input_feature_bk = input_enc_out_bk.last_hidden_state.to(device)
                    input_enc_out_aug1_bk = exp_enc(input_ids=input_ids_aug1, attention_mask=attention_mask_aug1)
                    input_feature_aug1_bk = input_enc_out_aug1_bk.last_hidden_state.to(device)
                    input_enc_out_aug2_bk = exp_enc(input_ids=input_ids_aug2, attention_mask=attention_mask_aug2)
                    input_feature_aug2_bk = input_enc_out_aug2_bk.last_hidden_state.to(device)
                    
                    # Calculate gate scores (alpha values)
                    gate_score_1 = gate_1.mean(dim=1).detach()
                    gate_score_2 = gate_2.mean(dim=1).detach()
                    gate_score_3 = gate_3.mean(dim=1).detach()
                    alpha_1 = (1-gate_score_1).mean()
                    alpha_2 = (1-gate_score_2).mean()
                    alpha_3 = (1-gate_score_3).mean()
                    
                    # Calculate InfoNCE losses
                    match_loss_orig = alpha_1 * info_nce_loss(input_feature_bk, exp_feature, temperature=0.1)
                    match_loss_aug1 = alpha_2 * info_nce_loss(input_feature_aug1_bk, exp_feature, temperature=0.1)
                    match_loss_aug2 = alpha_3 * info_nce_loss(input_feature_aug2_bk, exp_feature, temperature=0.1)
                    
                    k = 1/3
                    match_loss = (match_loss_orig + match_loss_aug1 + match_loss_aug2) * k
       
            # 版本特定的损失函数配置
            if args.model_version == 'v3':
                # 更保守的损失权重，避免训练不稳定
                loss = 0.7 * sup_loss + 0.2 * cons_loss + 0.1 * finegrain_loss
                if args.use_match_loss:
                    loss = loss + match_loss
            elif args.model_version == 'v5':
                # V5: Enhanced loss with explanation consistency regularization
                base_loss = 0.6 * sup_loss + 0.3 * cons_loss + 0.1 * finegrain_loss
                
                # Add explanation consistency loss if available from the model
                exp_consistency_loss = 0.0
                if hasattr(model.backbone.encoder, 'exp_consistency_loss'):
                    exp_consistency_loss = model.backbone.encoder.exp_consistency_loss
                    # Reset the loss after using it
                    delattr(model.backbone.encoder, 'exp_consistency_loss')
                
                # Adaptive loss weighting based on training progress
                epoch_factor = min(1.0, epoch / (args.n_epochs * 0.5))  # Ramp up over first half of training
                consistency_weight = 0.1 * epoch_factor
                
                loss = base_loss + consistency_weight * exp_consistency_loss
                
                # Add explanation-text alignment loss for better fusion
                # This encourages the model to use explanations more effectively
                if torch.rand(1).item() < 0.1:  # Sample 10% of batches for efficiency
                    # Get text and explanation representations from the encoder
                    with torch.no_grad():
                        text_only_outputs = model(input_ids=input_ids, attention_mask=attention_mask, exp_feature=None)
                        text_only_logits = text_only_outputs[1]  # Binary classification logits
                    
                    # Encourage consistency between text-only and text+explanation predictions
                    alignment_loss = F.kl_div(
                        F.log_softmax(out_labels_bi, dim=-1),
                        F.softmax(text_only_logits.detach(), dim=-1),
                        reduction='batchmean'
                    )
                    loss = loss + 0.05 * alignment_loss
            else:
                loss = sup_loss + cons_loss + finegrain_loss
                if args.use_match_loss:
                    loss = loss + match_loss

            # 版本3和v5添加NaN检测和修复
            if args.model_version in ['v3', 'v5']:
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN/Inf loss detected at batch {batch_idx}, epoch {epoch}")
                    # 创建一个小的有效损失值，而不是0
                    loss = torch.tensor(1e-6, device=device, requires_grad=True)
                    # 设置默认准确率并跳过这个batch
                    train_acc = 0.0
                    avg_acc.append(train_acc)
                    avg_loss.append(loss.item())
                    batch_idx = batch_idx + 1
                    continue
                
                if torch.isnan(out_labels).any() or torch.isnan(out_labels_bi).any():
                    print(f"Warning: NaN outputs detected at batch {batch_idx}, epoch {epoch}")
                    # 跳过这个batch，但要设置默认的train_acc
                    train_acc = 0.0
                    avg_acc.append(train_acc)
                    batch_idx = batch_idx + 1
                    continue
            
            # 添加损失调试信息
            if batch_idx == 0 and epoch == 0:
                print(f"Loss components - Sup: {sup_loss.item():.4f}, Cons: {cons_loss.item():.4f}, FG: {finegrain_loss.item():.4f}")
                if args.use_match_loss:
                    match_loss_value = match_loss.item() if hasattr(match_loss, 'item') else match_loss
                    print(f"Loss components - Match: {match_loss_value:.4f}")
                print(f"Model outputs - out_labels shape: {out_labels.shape}, out_labels_bi shape: {out_labels_bi.shape}")
                print(f"Output values - out_labels range: [{out_labels.min().item():.4f}, {out_labels.max().item():.4f}]")
                print(f"Output values - out_labels_bi range: [{out_labels_bi.min().item():.4f}, {out_labels_bi.max().item():.4f}]")
                print(f"Output values - out_labels has NaN: {torch.isnan(out_labels).any().item()}")
                print(f"Output values - out_labels_bi has NaN: {torch.isnan(out_labels_bi).any().item()}")
                print(f"Output values - out_labels has Inf: {torch.isinf(out_labels).any().item()}")
                print(f"Output values - out_labels_bi has Inf: {torch.isinf(out_labels_bi).any().item()}")

            optimizer.zero_grad()
            loss.backward()
            
            # 版本3添加梯度裁剪，更严格的设置
            if args.model_version == 'v3':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                if exp_enc is not None:
                    torch.nn.utils.clip_grad_norm_(exp_enc.parameters(), max_norm=0.5)
            
            avg_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
            _, pred = out_labels_bi.max(dim=-1)
            correct = pred.eq(targets).sum().item()
            train_acc = correct / len(targets)
            avg_acc.append(train_acc)
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss) if avg_loss else 0.0)
        train_accs.append(np.mean(avg_acc) if avg_acc else 0.0)
        
        # 使用epoch的平均准确率而不是最后一个batch的准确率
        epoch_train_acc = np.mean(avg_acc) if avg_acc else 0.0
        print("Iter {:03d} | Epoch {:05d} | Train Acc. {:.4f}".format(iter, epoch, epoch_train_acc))

        if epoch == n_epochs - 1:
            if exp_enc is not None:
                exp_enc.eval()
            model.eval()
            y_pred = []
            y_pred_res = []
            y_test = []

            for Batch_data in tqdm(test_loader):
                with torch.no_grad():
                    input_ids = Batch_data["input_ids"].to(device)
                    attention_mask = Batch_data["attention_mask"].to(device)
                    targets = Batch_data["labels"].to(device)
                    
                    if args.model_version in ['v4', 'v5']:
                        if args.use_sliding_window:
                            batch_size_curr = targets.size(0)
                            out_bi_list = []
                            for i in range(batch_size_curr):
                                text = Batch_data["news_text"][i]
                                explanation = Batch_data["explanation"][i]
                                _, val_out_i = sliding_window_infer_v4(model, tokenizer, text, explanation, args.window_size, args.window_stride, device, max_windows=args.eval_max_windows)
                                out_bi_list.append(val_out_i)
                            val_out = torch.cat(out_bi_list, dim=0)
                        else:
                            # For v4: use concatenated inputs (no token_type_ids)
                            text_boundary = Batch_data["text_boundary"]
                            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                                             text_boundary=text_boundary[0] if len(text_boundary) > 0 else None)
                            if len(model_outputs) == 3:
                                _, val_out, _ = model_outputs  # unpack 3 values when gate_values is returned
                            else:
                                _, val_out = model_outputs  # unpack 2 values normally
                    else:
                        # For v1, v2, v3: use exp_feature approach
                        if exp_enc is not None:
                            input_ids_exp = Batch_data["input_ids_exp"].to(device)
                            attention_mask_exp = Batch_data["attention_mask_exp"].to(device)
                            exp_enc_out = exp_enc(input_ids=input_ids_exp, attention_mask=attention_mask_exp)
                            exp_feature = exp_enc_out.last_hidden_state.to(device)
                        else:
                            exp_feature = None
                        model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, exp_feature=exp_feature,
                                              exp_attention_mask=attention_mask_exp)
                        if len(model_outputs) == 3:
                            _, val_out, _ = model_outputs  # unpack 3 values when gate_values is returned
                        else:
                            _, val_out = model_outputs  # unpack 2 values normally
                    
                    _, val_pred = val_out.max(dim=1)
                    y_pred.append(val_pred)
                    y_test.append(targets)

            y_pred = torch.cat(y_pred, dim=0)
            y_test = torch.cat(y_test, dim=0)

            acc = accuracy_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            precision, recall, fscore, _ = score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), average='macro')

            # Evaluate per emotion and compute averages
            emo_accs, emo_precs, emo_recs, emo_f1s = [], [], [], []
            emotion_results = {}  # Store individual emotion results
            for emo, x_test_res_e in emotion_tests.items():
                print(f"Evaluating Restyle emotion: {emo}")
                test_loader_res = create_eval_loader(x_test_res_e, y_test.detach().cpu().numpy(), z_test, tokenizer, exp_tokenizer, max_len, batch_size)
                y_pred_res = []
                for Batch_data in tqdm(test_loader_res):
                    with torch.no_grad():
                        input_ids_aug = Batch_data["input_ids"].to(device)
                        attention_mask_aug = Batch_data["attention_mask"].to(device)
                        if args.model_version in ['v4', 'v5']:
                            if args.use_sliding_window:
                                batch_size_curr = input_ids_aug.size(0)
                                out_bi_list = []
                                for i in range(batch_size_curr):
                                    text_aug = Batch_data["news_text"][i]
                                    explanation_aug = Batch_data["explanation"][i]
                                    _, val_out_aug_i = sliding_window_infer_v4(model, tokenizer, text_aug, explanation_aug, args.window_size, args.window_stride, device, max_windows=args.eval_max_windows)
                                    out_bi_list.append(val_out_aug_i)
                                val_out_aug = torch.cat(out_bi_list, dim=0)
                            else:
                                text_boundary_aug = Batch_data["text_boundary"]
                                model_outputs_aug = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug,
                                                   text_boundary=text_boundary_aug[0] if len(text_boundary_aug) > 0 else None)
                                if len(model_outputs_aug) == 3:
                                    _, val_out_aug, _ = model_outputs_aug  # unpack 3 values when gate_values is returned
                                else:
                                    _, val_out_aug = model_outputs_aug  # unpack 2 values normally
                        else:
                            if exp_enc is not None:
                                input_ids_exp = Batch_data["input_ids_exp"].to(device)
                                attention_mask_exp = Batch_data["attention_mask_exp"].to(device)
                                exp_enc_out = exp_enc(input_ids=input_ids_exp, attention_mask=attention_mask_exp)
                                exp_feature = exp_enc_out.last_hidden_state.to(device)
                            else:
                                exp_feature = None
                            model_outputs_aug = model(input_ids=input_ids_aug, attention_mask=attention_mask_aug, exp_feature=exp_feature,
                                                      exp_attention_mask=attention_mask_exp)
                            if len(model_outputs_aug) == 3:
                                _, val_out_aug, _ = model_outputs_aug  # unpack 3 values when gate_values is returned
                            else:
                                _, val_out_aug = model_outputs_aug  # unpack 2 values normally
                        _, val_pred_aug = val_out_aug.max(dim=1)
                        y_pred_res.append(val_pred_aug)
                y_pred_res = torch.cat(y_pred_res, dim=0)
                acc_e = accuracy_score(y_test.detach().cpu().numpy(), y_pred_res.detach().cpu().numpy())
                precision_e, recall_e, fscore_e, _ = score(y_test.detach().cpu().numpy(), y_pred_res.detach().cpu().numpy(), average='macro')
                emo_accs.append(acc_e); emo_precs.append(precision_e); emo_recs.append(recall_e); emo_f1s.append(fscore_e)
                
                # Store individual emotion results
                emotion_results[emo] = {
                    'acc': acc_e,
                    'precision': precision_e,
                    'recall': recall_e,
                    'f1': fscore_e
                }
                
                print(f"-----------------Restyle ({emo})-----------------")
                print(['Global Test Accuracy:{:.4f}'.format(acc_e),
                    'Precision:{:.4f}'.format(precision_e),
                    'Recall:{:.4f}'.format(recall_e),
                    'F1:{:.4f}'.format(fscore_e)])

            acc_res = float(np.mean(emo_accs)) if len(emo_accs) > 0 else 0.0
            precision_res = float(np.mean(emo_precs)) if len(emo_precs) > 0 else 0.0
            recall_res = float(np.mean(emo_recs)) if len(emo_recs) > 0 else 0.0
            fscore_res = float(np.mean(emo_f1s)) if len(emo_f1s) > 0 else 0.0


    checkpoint_path = build_checkpoint_path(datasetname, run_name, iter)
    checkpoint = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "exp_encoder_state_dict": exp_enc.state_dict() if exp_enc is not None else None,
        "args": vars(args),
        "dataset": datasetname,
        "run_name": run_name,
        "iteration": iter,
        "metrics": {
            "original": {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": fscore,
            },
            "restyle": {
                "accuracy": acc_res,
                "precision": precision_res,
                "recall": recall_res,
                "f1": fscore_res,
            },
            "emotions": emotion_results,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    print("-----------------End of Iter {:03d}-----------------".format(iter))
    print(['Global Test Accuracy:{:.4f}'.format(acc),
        'Precision:{:.4f}'.format(precision),
        'Recall:{:.4f}'.format(recall),
        'F1:{:.4f}'.format(fscore)])

    print("-----------------Restyle-----------------")
    print(['Global Test Accuracy:{:.4f}'.format(acc_res),
        'Precision:{:.4f}'.format(precision_res),
        'Recall:{:.4f}'.format(recall_res),
        'F1:{:.4f}'.format(fscore_res)])
    
    return acc, precision, recall, fscore, acc_res, precision_res, recall_res, fscore_res, emotion_results


datasetname=args.dataset_name
run_name = build_run_name(datasetname)
print(f"Run name: {run_name}")

# Keep checkpoint folder in sync with run naming convention.
checkpoint_run_dir = resolve_checkpoint_run_dir(datasetname, run_name)
checkpoint_run_dir.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint directory: {checkpoint_run_dir}")

batch_size = args.batch_size
max_len = 512
tokenizer = AutoTokenizer.from_pretrained(pretrained_backbone_name)
exp_tokenizer = AutoTokenizer.from_pretrained(pretrained_backbone_name)
n_epochs = args.n_epochs
iterations=args.iters

test_accs = []
prec_all, rec_all, f1_all = [], [], []
test_accs_res = []
prec_all_res, rec_all_res, f1_all_res = [], [], []

# Store emotion results across all iters
all_emotion_results = []


for iter in range(iterations):
    set_seed(iter)
    acc, prec, recall, f1, \
    acc_res, prec_res, recall_res, f1_res, emotion_results = train_model(tokenizer,
                                                exp_tokenizer,
                                                max_len,
                                                n_epochs,
                                                batch_size,
                                                datasetname,
                                                iter,
                                                run_name)

    test_accs.append(acc)
    prec_all.append(prec)
    rec_all.append(recall)
    f1_all.append(f1)
    test_accs_res.append(acc_res)
    prec_all_res.append(prec_res)
    rec_all_res.append(recall_res)
    f1_all_res.append(f1_res)
    all_emotion_results.append(emotion_results)

print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs) / iterations, sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))

print("Restyle_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
    sum(test_accs_res) / iterations, sum(prec_all_res) /iterations, sum(rec_all_res) /iterations, sum(f1_all_res) / iterations))

# Calculate and print detailed emotion statistics across all iters
print("\n" + "="*80)
print("DETAILED EMOTION STATISTICS ACROSS ALL ITERATIONS")
print("="*80)

if all_emotion_results:
    # Get all emotion names from the first iter
    emotion_names = list(all_emotion_results[0].keys()) if all_emotion_results[0] else []
    
    print(f"\nOriginal Test Results (Average across {iterations} iters):")
    print(f"Accuracy: {sum(test_accs) / iterations:.4f} ± {np.std(test_accs):.4f}")
    print(f"Precision: {sum(prec_all) / iterations:.4f} ± {np.std(prec_all):.4f}")
    print(f"Recall: {sum(rec_all) / iterations:.4f} ± {np.std(rec_all):.4f}")
    print(f"F1: {sum(f1_all) / iterations:.4f} ± {np.std(f1_all):.4f}")
    
    print(f"\nRestyle Test Results by Emotion (Average across {iterations} iters):")
    print("-" * 80)
    
    for emotion in emotion_names:
        emotion_accs = [iter_results[emotion]['acc'] for iter_results in all_emotion_results if emotion in iter_results]
        emotion_precs = [iter_results[emotion]['precision'] for iter_results in all_emotion_results if emotion in iter_results]
        emotion_recs = [iter_results[emotion]['recall'] for iter_results in all_emotion_results if emotion in iter_results]
        emotion_f1s = [iter_results[emotion]['f1'] for iter_results in all_emotion_results if emotion in iter_results]
        
        if emotion_accs:  # Only print if we have data for this emotion
            print(f"\n{emotion.upper()} Restyle:")
            print(f"  Accuracy:  {np.mean(emotion_accs):.4f} ± {np.std(emotion_accs):.4f}")
            print(f"  Precision: {np.mean(emotion_precs):.4f} ± {np.std(emotion_precs):.4f}")
            print(f"  Recall:    {np.mean(emotion_recs):.4f} ± {np.std(emotion_recs):.4f}")
            print(f"  F1:        {np.mean(emotion_f1s):.4f} ± {np.std(emotion_f1s):.4f}")
    
    print(f"\nOverall Restyle Average (Average across {iterations} iters):")
    print(f"Accuracy: {sum(test_accs_res) / iterations:.4f} ± {np.std(test_accs_res):.4f}")
    print(f"Precision: {sum(prec_all_res) / iterations:.4f} ± {np.std(prec_all_res):.4f}")
    print(f"Recall: {sum(rec_all_res) / iterations:.4f} ± {np.std(rec_all_res):.4f}")
    print(f"F1: {sum(f1_all_res) / iterations:.4f} ± {np.std(f1_all_res):.4f}")
    
    print(f"\nRobustness Gap (Original - Restyle):")
    print(f"Accuracy Gap: {(sum(test_accs) / iterations) - (sum(test_accs_res) / iterations):.4f}")
    print(f"F1 Gap: {(sum(f1_all) / iterations) - (sum(f1_all_res) / iterations):.4f}")

print("="*80)


summary_log_dir = Path("results")
summary_log_dir.mkdir(parents=True, exist_ok=True)
summary_log_path = summary_log_dir / f"log_{datasetname}_{args.model_name}.iter{iterations}"

with open(summary_log_path, 'a+') as f:
    f.write('-------------Original-------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs))
    f.write('All Prec.s:{}\n'.format(prec_all))
    f.write('All Rec.s:{}\n'.format(rec_all))
    f.write('All F1.s:{}\n'.format(f1_all))
    f.write('Average acc.: {} \n'.format(sum(test_accs) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all) /iterations, sum(rec_all) /iterations, sum(f1_all) / iterations))


    f.write('\n-------------Adversarial------------\n')
    f.write('All Acc.s:{}\n'.format(test_accs_res))
    f.write('All Prec.s:{}\n'.format(prec_all_res))
    f.write('All Rec.s:{}\n'.format(rec_all_res))
    f.write('All F1.s:{}\n'.format(f1_all_res))    
    f.write('Average acc.: {} \n'.format(sum(test_accs_res) / iterations))
    f.write('Average Prec / Rec / F1 (macro): {}, {}, {} \n'.format(sum(prec_all_res) /iterations, sum(rec_all_res) /iterations, sum(f1_all_res) / iterations))
    
    # Write detailed emotion statistics to log file
    if all_emotion_results:
        f.write('\n' + '='*80 + '\n')
        f.write('DETAILED EMOTION STATISTICS ACROSS ALL ITERATIONS\n')
        f.write('='*80 + '\n')
        
        emotion_names = list(all_emotion_results[0].keys()) if all_emotion_results[0] else []
        
        f.write(f'\nOriginal Test Results (Average across {iterations} iters):\n')
        f.write(f'Accuracy: {sum(test_accs) / iterations:.4f} ± {np.std(test_accs):.4f}\n')
        f.write(f'Precision: {sum(prec_all) / iterations:.4f} ± {np.std(prec_all):.4f}\n')
        f.write(f'Recall: {sum(rec_all) / iterations:.4f} ± {np.std(rec_all):.4f}\n')
        f.write(f'F1: {sum(f1_all) / iterations:.4f} ± {np.std(f1_all):.4f}\n')
        
        f.write(f'\nRestyle Test Results by Emotion (Average across {iterations} iters):\n')
        f.write('-' * 80 + '\n')
        
        for emotion in emotion_names:
            emotion_accs = [iter_results[emotion]['acc'] for iter_results in all_emotion_results if emotion in iter_results]
            emotion_precs = [iter_results[emotion]['precision'] for iter_results in all_emotion_results if emotion in iter_results]
            emotion_recs = [iter_results[emotion]['recall'] for iter_results in all_emotion_results if emotion in iter_results]
            emotion_f1s = [iter_results[emotion]['f1'] for iter_results in all_emotion_results if emotion in iter_results]
            
            if emotion_accs:
                f.write(f'\n{emotion.upper()} Restyle:\n')
                f.write(f'  Accuracy:  {np.mean(emotion_accs):.4f} ± {np.std(emotion_accs):.4f}\n')
                f.write(f'  Precision: {np.mean(emotion_precs):.4f} ± {np.std(emotion_precs):.4f}\n')
                f.write(f'  Recall:    {np.mean(emotion_recs):.4f} ± {np.std(emotion_recs):.4f}\n')
                f.write(f'  F1:        {np.mean(emotion_f1s):.4f} ± {np.std(emotion_f1s):.4f}\n')
        
        f.write(f'\nOverall Restyle Average (Average across {iterations} iters):\n')
        f.write(f'Accuracy: {sum(test_accs_res) / iterations:.4f} ± {np.std(test_accs_res):.4f}\n')
        f.write(f'Precision: {sum(prec_all_res) / iterations:.4f} ± {np.std(prec_all_res):.4f}\n')
        f.write(f'Recall: {sum(rec_all_res) / iterations:.4f} ± {np.std(rec_all_res):.4f}\n')
        f.write(f'F1: {sum(f1_all_res) / iterations:.4f} ± {np.std(f1_all_res):.4f}\n')
        
        f.write(f'\nRobustness Gap (Original - Restyle):\n')
        f.write(f'Accuracy Gap: {(sum(test_accs) / iterations) - (sum(test_accs_res) / iterations):.4f}\n')
        f.write(f'F1 Gap: {(sum(f1_all) / iterations) - (sum(f1_all_res) / iterations):.4f}\n')
        
        f.write('='*80 + '\n')
