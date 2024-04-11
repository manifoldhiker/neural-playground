import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_lens import HookedTransformer, HookedTransformerConfig

from .tree import list2tree
from .tree_dataset import TreeDataset, parse_input_idx, input_tokens_to_tree, tree_to_edges


def accuracy_by_depth(outputs, input_idx, out_mask):
    metrics = []
    
    for pred_row, gt_row, row_mask in zip(outputs.argmax(dim=-1), input_idx[:, 1:], out_mask):
        is_path_correct = (pred_row[row_mask] == gt_row[row_mask]).all()
        depth = row_mask.sum()
    
        metrics.append({'is_path_correct': is_path_correct.item(), 'depth': depth.item()})
    
    
    accuracy_by_goal_depth = pd.DataFrame(metrics).groupby('depth')['is_path_correct'].mean().to_dict()


    accuracy_by_goal_depth = {f'acc/depth={d}':v for d,v in accuracy_by_goal_depth.items()}
    
    return accuracy_by_goal_depth


class TreeTrainer:
    def __init__(self, conf):
        dataset = TreeDataset(conf.n_nodes)
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=conf['batch_size'])

        conf.model["n_ctx"] = dataset.tokenizer.MAX_SEQ_LEN
        conf.model["d_vocab"] = len(dataset.tokenizer.token2idx)
        
        model_cfg = HookedTransformerConfig(
            **conf.model
        )
        
        device = conf.device
        self.device = device

        self.model = HookedTransformer(model_cfg).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **conf.optimizer)


    def train_step(self, batch):
        input_idx = batch['input_idx'].to(self.device)
        mask = batch['task_mask'].to(self.device)
        
        inputs = input_idx[:, :-1]
        
        out_mask = mask[:, 1:]
        targets = input_idx[:, 1:][out_mask]
        
        
        # print(input_idx[:1, :4])
        outputs = self.model(inputs)
        
        predictions = outputs[out_mask]
        
        loss = F.cross_entropy(predictions, targets)

        is_correct = (predictions.argmax(dim=-1) == targets)
        accuracy_mean = is_correct.float().mean()
        metrics = accuracy_by_depth(outputs, input_idx, out_mask)
        metrics['accuracy/mean'] = accuracy_mean.item()

        return loss, metrics

    def tok(self, *args, **kwargs): return self.dataset.tokenizer.tokenize(*args, **kwargs)
    def detok(self, *args, **kwargs): return self.dataset.tokenizer.detokenize(*args, **kwargs)


    def inference_on_prompt(trainer, prompt):
        tokens = trainer.tok(prompt)
        tokens = torch.tensor(tokens)[None]
        pred_token_greedy = trainer.model(tokens)[0, -1].argmax()
        pred_token = trainer.detok([pred_token_greedy])
        return pred_token[0]
    
    
    def print_sample_pred(trainer, sample_input_idx):
        """
            sample_input_idx - [seq_len]
        """
        ROOT_DELIM_TOKEN_IDX = trainer.tok([':'])[0]
    
    
        upper_task_bound = sample_input_idx.tolist().index(ROOT_DELIM_TOKEN_IDX) + 2
        prompt_autoregressive = trainer.detok(sample_input_idx)[:upper_task_bound]
        input_tree = input_tokens_to_tree(prompt_autoregressive)
        
        prompt = prompt_autoregressive
        
        parsed_input = parse_input_idx(sample_input_idx, trainer.dataset.tokenizer)
        gt_path = parsed_input['path']
        pred_path = []
        
        for i in range(len(gt_path)):
            pred_token = trainer.inference_on_prompt(prompt)
            pred_path.append(pred_token)
            prompt += [pred_token]
        
        accuracy = (np.array(gt_path) == np.array(pred_path)).astype(float).mean()
        print('*'*100)
        print(input_tree)
        print()
        print(f'goal={parsed_input["goal"]}')
        print(f'{accuracy=} {gt_path=} {pred_path=}' )
        print('*'*100)
