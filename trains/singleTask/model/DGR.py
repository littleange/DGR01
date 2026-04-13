import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder

class ComplementaryInformationExtractor(nn.Module):
    def __init__(self, query_dim, key_dim, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.inner_dim = self.head_dim * self.num_heads 
        
        self.q_proj = nn.Linear(query_dim, self.inner_dim)
        self.k_proj = nn.Linear(key_dim, self.inner_dim)
        self.v_proj = nn.Linear(key_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)
        self.temperature = nn.Parameter(torch.tensor(self.head_dim ** -0.5)) 
        self.mode_factor = nn.Parameter(torch.tensor(1.0)) 

        self.dropout = nn.Dropout(dropout)

    def forward(self, shared_query, private_key_value):
        B, N, _ = private_key_value.shape
        
        if shared_query.dim() == 2:
            shared_query = shared_query.unsqueeze(1)
        Q = self.q_proj(shared_query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2) 
        K = self.k_proj(private_key_value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) 
        V = self.v_proj(private_key_value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) 

        attn_logits = (Q @ K.transpose(-2, -1)) * self.temperature
        adaptive_logits = attn_logits * self.mode_factor

        attn_weights = F.softmax(adaptive_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ V).transpose(1, 2).reshape(B, 1, self.inner_dim) 
        output = self.out_proj(context)
        
        return output.squeeze(1), attn_weights
class RCR_Module(nn.Module):
    def __init__(self, fused_dim, private_dim, d_model=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model if d_model is not None else fused_dim
        
        self.cie_extractor = ComplementaryInformationExtractor(
            query_dim=fused_dim, 
            key_dim=private_dim, 
            embed_dim=self.d_model, 
            dropout=dropout
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(fused_dim + self.d_model, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, 1),
            nn.Sigmoid() 
        )
        
        self.layer_norm = nn.LayerNorm(self.d_model)
       
        self.match_proj = nn.Linear(self.d_model, fused_dim) if self.d_model != fused_dim else nn.Identity()

        self._init_gate_to_low_value(target_value=0.1)

    def _init_gate_to_low_value(self, target_value=0.1):
        bias_value = math.log(target_value / (1.0 - target_value))
        
        last_layer = self.gate_net[-2] 
        nn.init.zeros_(last_layer.weight) 
        nn.init.constant_(last_layer.bias, bias_value) 

    def forward(self, fused_shared, private_feats_list):
        p_concat = torch.cat(private_feats_list, dim=2).transpose(1, 2)
        comp_info, _ = self.cie_extractor(fused_shared, p_concat)
        combined = torch.cat([fused_shared, comp_info], dim=-1)
        gate_val = self.gate_net(combined) 
        
        comp_info_proj = self.match_proj(comp_info)
        refined_feature = fused_shared + gate_val * comp_info_proj
        
        return self.layer_norm(refined_feature), gate_val
class HybridModalityGate(nn.Module):
    def __init__(self, dim_l, dim_a, dim_v, temperature=1.0):
        super(HybridModalityGate, self).__init__()
        self.T = temperature
        combined_dim = dim_l + dim_a + dim_v
        self.context_net = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(combined_dim // 2, 3) 
        )
        
        self.energy_scale = nn.Parameter(torch.tensor(0.0))

    def energy_score(self, logits):
        if logits.dim() > 1 and logits.shape[-1] > 1:
            energy = -self.T * torch.logsumexp(logits / self.T, dim=-1)
        else:
            energy = logits.squeeze(-1)
        return energy.unsqueeze(1) 

    def forward(self, h_l, h_a, h_v, logits_l, logits_a, logits_v):
        context_feat = torch.cat([h_l, h_a, h_v], dim=-1)
        base_logits = self.context_net(context_feat) 
        u_l = self.energy_score(logits_l)
        u_a = self.energy_score(logits_a)
        u_v = self.energy_score(logits_v)
        energies = torch.cat([u_l, u_a, u_v], dim=-1) 
        rectified_logits = base_logits - (torch.abs(self.energy_scale) * energies)
        
        weights = F.softmax(rectified_logits, dim=-1)
        
        return weights, (u_l, u_a, u_v), rectified_logits
class DGR(nn.Module):
    def __init__(self, args):
        super(DGR, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        
        
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        
        
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        output_dim = 1 

        
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        
        self.enc_private_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, bias=False)
        self.enc_private_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, bias=False)
        self.enc_private_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, bias=False)

        
        self.enc_shared_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, bias=False)
        self.enc_shared_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, bias=False)
        self.enc_shared_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, bias=False)

        
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, bias=False)

       
        self.trans_l_with_a = self.get_network(self_type='la', layers=self.layers)
        self.trans_l_with_v = self.get_network(self_type='lv', layers=self.layers)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        combined_dim_high = self.d_l 
        
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim) 
        
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim) 
        
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim) 

        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)

        self.modality_gate = HybridModalityGate(self.d_l, self.d_a, self.d_v, temperature=1.0)
        
        self.fusion_dim = self.d_l + self.d_a + self.d_v 
        
        self.rcr_module = RCR_Module(
            fused_dim=self.fusion_dim, 
            private_dim=self.d_l,      
            d_model=self.fusion_dim,   
            dropout=self.attn_dropout
        )

        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, 128)
        )
        self.proj1 = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.proj2 = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.out_layer = nn.Linear(self.fusion_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl', 'l_mem']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va', 'a_mem']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av', 'v_mem']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim, num_heads=self.num_heads, layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout, relu_dropout=self.relu_dropout, res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout, attn_mask=self.attn_mask)

    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)
        
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
                
        p_l = self.enc_private_l(proj_x_l)
        p_a = self.enc_private_a(proj_x_a)
        p_v = self.enc_private_v(proj_x_v)
        s_l_feat = self.enc_shared_l(proj_x_l)
        s_a_feat = self.enc_shared_a(proj_x_a)
        s_v_feat = self.enc_shared_v(proj_x_v)
        rec_input_l = torch.cat([s_l_feat, p_l], dim=1)
        rec_input_a = torch.cat([s_a_feat, p_a], dim=1)
        rec_input_v = torch.cat([s_v_feat, p_v], dim=1)
        recon_l = self.decoder_l(rec_input_l)
        recon_a = self.decoder_a(rec_input_a)
        recon_v = self.decoder_v(rec_input_v)
        s_l = s_l_feat.permute(2, 0, 1)
        s_a = s_a_feat.permute(2, 0, 1)
        s_v = s_v_feat.permute(2, 0, 1)

        
        def _unpack_last(h):
            if isinstance(h, tuple): h = h[0]
            try: return h[-1]
            except Exception: return h
        def _to_vector(t):
            if t is None: return None
            if t.dim() == 3: return t[-1] 
            elif t.dim() == 2: return t
            else: raise RuntimeError(f"Unexpected shape: {t.shape}")

        last_h_l = (_to_vector(_unpack_last(self.trans_l_with_a(s_l, s_a, s_a))) + 
                    _to_vector(_unpack_last(self.trans_l_with_v(s_l, s_v, s_v))) + 
                    _to_vector(_unpack_last(self.trans_l_mem(s_l)))) / 3.0
        last_h_a = (_to_vector(_unpack_last(self.trans_a_with_l(s_a, s_l, s_l))) + 
                    _to_vector(_unpack_last(self.trans_a_with_v(s_a, s_v, s_v))) + 
                    _to_vector(_unpack_last(self.trans_a_mem(s_a)))) / 3.0
        last_h_v = (_to_vector(_unpack_last(self.trans_v_with_l(s_v, s_l, s_l))) + 
                    _to_vector(_unpack_last(self.trans_v_with_a(s_v, s_a, s_a))) + 
                    _to_vector(_unpack_last(self.trans_v_mem(s_v)))) / 3.0

        
        hs_proj_l = self.proj2_l_high(F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l += last_h_l
        logits_l = self.out_layer_l_high(hs_proj_l)
        
        hs_proj_v = self.proj2_v_high(F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v += last_h_v
        logits_v = self.out_layer_v_high(hs_proj_v)
        
        hs_proj_a = self.proj2_a_high(F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a += last_h_a
        logits_a = self.out_layer_a_high(hs_proj_a)

        
        feat_l = torch.sigmoid(self.projector_l(hs_proj_l))
        feat_v = torch.sigmoid(self.projector_v(hs_proj_v))
        feat_a = torch.sigmoid(self.projector_a(hs_proj_a))

        gate_weights, energies, raw_logits = self.modality_gate(
            feat_l, feat_a, feat_v, 
            logits_l, logits_a, logits_v
        )
        
        w_l, w_v, w_a = gate_weights[:, 0:1], gate_weights[:, 1:2], gate_weights[:, 2:3]

        weighted_l = feat_l * w_l
        weighted_v = feat_v * w_v
        weighted_a = feat_a * w_a
        
        fused_hs = torch.cat([weighted_l, weighted_a, weighted_v], dim=1)
        private_feats_list = [p_l, p_a, p_v] 
        refined_hs, rcr_gate_val = self.rcr_module(fused_hs, private_feats_list)

        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(refined_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += refined_hs
        output = self.out_layer(last_hs_proj)
        contrastive_feat = self.contrastive_head(refined_hs)
        contrastive_feat = F.normalize(contrastive_feat, dim=1)

        res = {
            'logits_l_hetero': logits_l,
            'logits_v_hetero': logits_v,
            'logits_a_hetero': logits_a,
            'output_logit': output,
            'c_l': s_l_feat,  
            'c_a': s_a_feat,
            'c_v': s_v_feat,

            'p_l': p_l,        
            'p_a': p_a,
            'p_v': p_v,
            'contrastive_feat': contrastive_feat,
            'gate_weights': gate_weights,
            'rcr_gate_val': rcr_gate_val, 
            'qmf_info': {
                'raw_weights': raw_logits,
                'energies': energies 
            },
            'decomp_info': {
                'original': [proj_x_l, proj_x_a, proj_x_v], 
                'recon': [recon_l, recon_a, recon_v],       
                'shared': [s_l_feat, s_a_feat, s_v_feat],   
                'private': [p_l, p_a, p_v]                  
            }
        }
        return res