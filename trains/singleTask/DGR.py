import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        features = F.normalize(features, dim=1)
        device = features.device
        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = - mean_log_prob_pos
        return loss.mean()

class DGR():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss() 
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.scl_loss = SupConLoss(temperature=0.1)
        self.gamma_aux = 10 #MOSEI 0.01 MOSI 10
        self.gamma_decomp = 0.1  
        self.gamma_scl = 0.1 #MOSEI 0.01 MOSI 0.1
        self.gamma_reg = 0.1    
        self.lambda_rec = 1.0
        self.lambda_cyc = 0.1
        self.lambda_mar = 0.1
        self.lambda_orth = 0.1

    def do_train(self, model, dataloader, return_epoch_results=False):
        net_DGR = model[0]
        params = net_DGR.parameters()
        optimizer = optim.AdamW(params, lr=self.args.learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {'train': [], 'valid': [], 'test': []}
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true = [], []
            net_DGR.train()
            loss_meters = {
                'Total': 0.0, 'Task': 0.0, 'Aux': 0.0, 'SCL': 0.0, 'Reg': 0.0,
                'Decomp': 0.0, 'Rec': 0.0, 'Cyc': 0.0, 'Mar': 0.0, 'Orth': 0.0
            }
            
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    # Data
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)
                    # Forward
                    output = net_DGR(text, audio, vision)
                    loss_main = self.criterion(output['output_logit'], labels)
                    loss_aux = (self.criterion(output['logits_l_hetero'], labels) + 
                                self.criterion(output['logits_a_hetero'], labels) + 
                                self.criterion(output['logits_v_hetero'], labels))
                    
                    task_loss_total = loss_main + self.gamma_aux * loss_aux
                    loss_scl = torch.tensor(0.0, device=self.args.device)
                    if 'contrastive_feat' in output:
                        loss_scl = self.scl_loss(output['contrastive_feat'], labels)
                    loss_decomp_total = torch.tensor(0.0, device=self.args.device)
                    l_rec, l_cyc, l_mar, l_orth = [torch.tensor(0.0, device=self.args.device)] * 4
                    
                    if 'decomp_info' in output:
                        info = output['decomp_info']
                        mse = nn.MSELoss()
                        if info['recon'][0] is None:
                            l_rec = 0.0
                        else:
                            l_rec = sum([mse(r, o) for r, o in zip(info['recon'], info['original'])])
                        
                        for s, p in zip(info['shared'], info['private']):
                            l_orth += torch.mean(torch.abs(F.cosine_similarity(s.view(s.size(0), -1), p.view(p.size(0), -1), dim=1)))
                        
                        private_encoders = [net_DGR.enc_private_l, net_DGR.enc_private_a, net_DGR.enc_private_v]
                        for i, recon_feat in enumerate(info['recon']):
                            l_cyc += mse(private_encoders[i](recon_feat), info['private'][i])
                        
                        margin_val = 0.2
                        for s_feat in info['shared']:
                            s_vec = F.normalize(s_feat.mean(dim=-1), p=2, dim=1)
                            label_diff = torch.abs(labels - labels.t())
                            is_pos = (label_diff < 1.0) & (~torch.eye(labels.size(0), device=labels.device).bool())
                            is_neg = (label_diff >= 1.0)
                            if is_pos.sum() > 0 and is_neg.sum() > 0:
                                pos = (s_vec @ s_vec.t() * is_pos.float()).sum(1) / (is_pos.float().sum(1) + 1e-6)
                                neg = (s_vec @ s_vec.t() * is_neg.float()).sum(1) / (is_neg.float().sum(1) + 1e-6)
                                l_mar += F.relu(margin_val - pos + neg).mean()

                        loss_decomp_total = self.lambda_rec * l_rec + self.lambda_cyc * l_cyc + \
                                            self.lambda_mar * l_mar + self.lambda_orth * l_orth

                    loss_reg = torch.tensor(0.0, device=self.args.device)
                    if 'qmf_info' in output:
                        cur_error = torch.abs(output['output_logit'] - labels).detach()
                        weights = output['gate_weights']
                        for m_idx in range(weights.shape[1]):
                            w = weights[:, m_idx:m_idx+1]
                            diff_w = w - w.t()
                            diff_err = cur_error - cur_error.t()
                            is_harder = (diff_err > 0).float()
                            loss_reg += (F.relu(diff_w) * is_harder).mean()
                    combined_loss = task_loss_total + \
                                    self.gamma_scl * loss_scl + \
                                    self.gamma_decomp * loss_decomp_total + \
                                    self.gamma_reg * loss_reg

                    combined_loss.backward()
                    
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)
                    loss_meters['Total'] += combined_loss.item()
                    loss_meters['Task'] += loss_main.item()
                    loss_meters['Aux'] += loss_aux.item()
                    loss_meters['SCL'] += loss_scl.item()
                    loss_meters['Reg'] += loss_reg.item()
                    loss_meters['Decomp'] += loss_decomp_total.item()
                    loss_meters['Rec'] += l_rec.item()
                    loss_meters['Cyc'] += l_cyc.item()
                    loss_meters['Mar'] += l_mar.item()
                    loss_meters['Orth'] += l_orth.item()
                    
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
            
            if not left_epochs:
                optimizer.step()
            scheduler.step()
            num_batches = len(dataloader['train'])
            for k in loss_meters:
                loss_meters[k] /= num_batches

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            log_msg = (
                f">> Epoch: {epochs} TRAIN-({self.args.model_name})\n"
                f"   [Main] Total: {loss_meters['Total']:.4f} | Task: {loss_meters['Task']:.4f} | "
                f"MAE/Acc: {dict_to_str(train_results)}\n"
                f"   [Subs] Aux: {loss_meters['Aux']:.4f} | SCL: {loss_meters['SCL']:.4f} | Reg: {loss_meters['Reg']:.4f} | "
                f"Decomp: {loss_meters['Decomp']:.4f} (R:{loss_meters['Rec']:.2f}, C:{loss_meters['Cyc']:.2f}, M:{loss_meters['Mar']:.2f}, O:{loss_meters['Orth']:.2f})"
            )
            logger.info(log_msg)
            val_results = self.do_test(net_DGR, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(net_DGR.state_dict(), self.args.model_save_path)
                logger.info(f"*** Best Model Saved at Epoch {epochs} ***")
            
            if isBetter or epochs % 5 == 0:
                test_results = self.do_test(net_DGR, dataloader['test'], mode="TEST")
                if return_epoch_results:
                    epoch_results['test'].append(test_results)
            
            if return_epoch_results:
                train_results["Loss"] = loss_meters['Total'] 
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
            min_epochs = 30
            if epochs > min_epochs and (epochs - best_epoch >= self.args.early_stop):
                logger.info(f"Early stopping at epoch {epochs}")
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader, desc=mode, leave=False) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)
                    
                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
        
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        return eval_results