import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedDimWise(nn.Module):
    def __init__(self, window, n_sensors, patch_size, d_model, pos_learnable=True):
        super().__init__()
        self.P = patch_size
        self.d_model = d_model
        self.n_patches = math.ceil(window / patch_size)
        self.patch_proj = nn.Linear(self.P, d_model, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(n_sensors, self.n_patches, d_model)) if pos_learnable else None
        if pos_learnable: nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, W, S = x.shape
        pad_len = (self.n_patches * self.P) - W
        if pad_len > 0: x = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
        x = x.view(B, self.n_patches, self.P, S).permute(0, 3, 1, 2).contiguous()
        emb = self.patch_proj(x.view(B * S * self.n_patches, self.P)).view(B, S, self.n_patches, self.d_model)
        if self.pos_embed is not None: emb = emb + self.pos_embed.unsqueeze(0)
        return emb

class STARAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.temporal_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.sensor_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.sensor_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, T, d = x.shape
        x_flat = x.view(B * S, T, d)
        temp_out, _ = self.temporal_mha(x_flat, x_flat, x_flat)
        x = self.temporal_norm(x + temp_out.view(B, S, T, d))
        
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * T, S, d)
        sensor_out, _ = self.sensor_mha(x_flat, x_flat, x_flat)
        x = self.sensor_norm(x + sensor_out.view(B, T, S, d).permute(0, 2, 1, 3))
        
        return self.final_norm(x + self.ffn(x))

class PatchMerging(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)
    def forward(self, x):
        B, S, T, d = x.shape
        if T <= 1: return x
        if T % 2 == 1: x, T = x[:, :, :-1, :], T - 1
        return self.proj(torch.cat([x[:, :, 0::2, :], x[:, :, 1::2, :]], dim=-1))

class STAREncoder(nn.Module):
    def __init__(self, n_scales, d_model, nhead, ffn_dim, dropout, n_layers_per_scale=4):
        super().__init__()
        self.n_scales = n_scales
        self.layers = nn.ModuleList([nn.ModuleList([STARAttentionBlock(d_model, nhead, ffn_dim, dropout) for _ in range(n_layers_per_scale)]) for _ in range(n_scales)])
        self.patch_merging = nn.ModuleList([PatchMerging(d_model) for _ in range(n_scales - 1)])

    def forward(self, x):
        features, cur = [], x
        for i in range(self.n_scales):
            for layer in self.layers[i]: cur = layer(cur)
            features.append(cur)
            if i < self.n_scales - 1: cur = self.patch_merging[i](cur)
        return features

class DecoderBlockTwoStage(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim=256, dropout=0.05):
        super().__init__()
        self.temporal_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.sensor_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, dec, enc_kv):
        B, S, T_dec, d = dec.shape
        dec_temp_in = dec.reshape(B * S, T_dec, d)
        temp_out, _ = self.temporal_mha(dec_temp_in, dec_temp_in, dec_temp_in)
        dec_after_temp = temp_out.view(B, S, T_dec, d)
        
        dec_sensor_in = dec_after_temp.permute(0, 2, 1, 3).reshape(B * T_dec, S, d)
        sensor_out, _ = self.sensor_mha(dec_sensor_in, dec_sensor_in, dec_sensor_in)
        dec = self.self_attn_norm(dec + sensor_out.view(B, T_dec, S, d).permute(0, 2, 1, 3))

        cross_out, _ = self.cross_mha(dec.reshape(B, S * T_dec, d), enc_kv, enc_kv)
        dec = self.cross_attn_norm(dec + cross_out.view(B, S, T_dec, d))
        
        return self.ffn_norm(dec + self.ffn(dec))

class STARDecoder(nn.Module):
    def __init__(self, n_scales, d_model, nhead, ffn_dim, dropout, n_layers_per_scale=2):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleList([DecoderBlockTwoStage(d_model, nhead, ffn_dim, dropout) for _ in range(n_layers_per_scale)]) for _ in range(n_scales)])

class PredictionHead(nn.Module):
    def __init__(self, d_model, ffn_dim, n_scales, dropout):
        super().__init__()
        self.scale_mlps = nn.ModuleList([nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)) for _ in range(n_scales)])
        self.final_mlp = nn.Sequential(nn.Linear(d_model * n_scales, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, 1))

    def forward(self, dec_outputs):
        pooled = [self.scale_mlps[i](f.mean(dim=(1, 2))) for i, f in enumerate(dec_outputs)]
        return self.final_mlp(torch.cat(pooled, dim=-1)).view(-1)

class STARModelFull(nn.Module):
    def __init__(self, window, n_sensors, d_model, nhead, num_scales, ffn_dim=256, patch_size=4, dropout=0.25, encoder_layers_per_scale=4, decoder_layers_per_scale=2, pos_learnable=True):
        super().__init__()
        self.patch_embed = PatchEmbedDimWise(window, n_sensors, patch_size, d_model, pos_learnable)
        self.encoder = STAREncoder(num_scales, d_model, nhead, ffn_dim, dropout, encoder_layers_per_scale)
        self.decoder = STARDecoder(num_scales, d_model, nhead, ffn_dim, dropout, decoder_layers_per_scale)
        self.pred_head = PredictionHead(d_model, ffn_dim, num_scales, dropout)

    def forward(self, x):
        enc_feats = self.encoder(self.patch_embed(x))
        dec_outs = []
        dec_input = None

        for i in reversed(range(len(enc_feats))):
            enc_feat = enc_feats[i]
            B, S, T, d = enc_feat.shape
            
            if dec_input is None: dec_input = enc_feat
            else:
                _, _, T_prev, _ = dec_input.shape
                if T > T_prev:
                    dec_input = dec_input.repeat_interleave(T // T_prev, dim=2)
                    if dec_input.shape[2] != T:
                        dec_input = F.interpolate(dec_input.permute(0,1,3,2), size=T, mode='linear', align_corners=False).permute(0,1,3,2)

            cur = dec_input
            enc_kv_current = enc_feat.view(B, S * T, d)
            for blk in self.decoder.blocks[i]: cur = blk(cur, enc_kv_current)
            dec_outs.append(cur)
            dec_input = cur

        return self.pred_head(dec_outs[::-1])
