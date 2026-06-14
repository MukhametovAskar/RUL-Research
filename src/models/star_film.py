import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(max_len, d_model, device):
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class PatchEmbedDimWise(nn.Module):
    def __init__(self, window, n_sensors, patch_size, d_model, pos_learnable=True):
        super().__init__()
        self.window = window
        self.P = patch_size
        self.n_patches = math.ceil(window / patch_size)
        self.d_model = d_model
        self.patch_proj = nn.Linear(self.P, d_model, bias=True)
        if pos_learnable:
            self.pos_embed = nn.Parameter(torch.zeros(n_sensors, self.n_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(self, x):
        B, W, S = x.shape
        pad_len = (self.n_patches * self.P) - W
        if pad_len > 0:
            x = torch.cat([x, x[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
        x = x.view(B, self.n_patches, self.P, S).permute(0, 3, 1, 2).contiguous()
        emb_flat = self.patch_proj(x.view(B * S * self.n_patches, self.P))
        emb = emb_flat.view(B, S, self.n_patches, self.d_model)
        if self.pos_embed is not None:
            emb = emb + self.pos_embed.unsqueeze(0)
        return emb


class STARAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.temporal_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.temporal_norm1 = nn.LayerNorm(d_model)
        self.temporal_ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.temporal_norm2 = nn.LayerNorm(d_model)

        self.sensor_mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.sensor_norm1 = nn.LayerNorm(d_model)
        self.sensor_ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.sensor_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, T, d = x.shape
        res = x
        x_flat = x.view(B * S, T, d)
        temp_out, _ = self.temporal_mha(x_flat, x_flat, x_flat)
        x = self.temporal_norm1(res + temp_out.view(B, S, T, d))
        x = self.temporal_norm2(x + self.temporal_ffn(x))

        res = x
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * T, S, d)
        sensor_out, _ = self.sensor_mha(x_flat, x_flat, x_flat)
        x = self.sensor_norm1(res + sensor_out.view(B, T, S, d).permute(0, 2, 1, 3))
        x = self.sensor_norm2(x + self.sensor_ffn(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        B, S, T, d = x.shape
        if T <= 1:
            return x
        if T % 2 == 1:
            x = x[:, :, :-1, :]
            T = T - 1
        left = x[:, :, 0::2, :]
        right = x[:, :, 1::2, :]
        merged = torch.cat([left, right], dim=-1)
        return self.proj(merged)


class STAREncoder(nn.Module):
    def __init__(self, n_scales, d_model, nhead, ffn_dim, dropout, n_layers_per_scale=3):
        super().__init__()
        self.n_scales = n_scales
        self.layers = nn.ModuleList([
            nn.ModuleList([STARAttentionBlock(d_model, nhead, ffn_dim, dropout) for _ in range(n_layers_per_scale)])
            for _ in range(n_scales)
        ])
        self.patch_merging = nn.ModuleList([PatchMerging(d_model) for _ in range(n_scales - 1)])

    def forward(self, x):
        features = []
        cur = x
        for i in range(self.n_scales):
            for layer in self.layers[i]:
                cur = layer(cur)
            features.append(cur)
            if i < self.n_scales - 1:
                cur = self.patch_merging[i](cur)
        return features


class DecoderBlockTwoStage(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim=256, dropout=0.05):
        super().__init__()
        self.temporal_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.sensor_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.sensor_norm = nn.LayerNorm(d_model)
        self.cross_mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, dec, enc_feat, temporal_causal_mask=None):
        B, S, T_dec, d = dec.shape
        _, _, T_enc, _ = enc_feat.shape

        res = dec
        dec_temp_in = dec.reshape(B * S, T_dec, d)
        temp_out, _ = self.temporal_mha(dec_temp_in, dec_temp_in, dec_temp_in, attn_mask=temporal_causal_mask)
        dec = self.temporal_norm(res + temp_out.reshape(B, S, T_dec, d))

        res = dec
        dec_sensor_in = dec.permute(0, 2, 1, 3).reshape(B * T_dec, S, d)
        sensor_out, _ = self.sensor_mha(dec_sensor_in, dec_sensor_in, dec_sensor_in)
        dec = self.sensor_norm(res + sensor_out.reshape(B, T_dec, S, d).permute(0, 2, 1, 3))

        res = dec
        dec_cross_in = dec.reshape(B, S * T_dec, d)
        enc_cross_kv = enc_feat.reshape(B, S * T_enc, d)
        cross_out, _ = self.cross_mha(dec_cross_in, enc_cross_kv, enc_cross_kv)
        dec = self.cross_norm(res + cross_out.reshape(B, S, T_dec, d))

        res = dec
        dec = self.ffn_norm(res + self.ffn(dec))
        return dec


class STARDecoder(nn.Module):
    def __init__(self, n_scales, d_model, nhead, ffn_dim, dropout, n_layers_per_scale=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([DecoderBlockTwoStage(d_model, nhead, ffn_dim, dropout) for _ in range(n_layers_per_scale)])
            for _ in range(n_scales)
        ])

    def forward(self, dec_in, enc_kv, blocks_for_scale):
        cur = dec_in
        for blk in blocks_for_scale:
            cur = blk(cur, enc_kv, temporal_causal_mask=None)
        return cur


class PredictionHead(nn.Module):
    def __init__(self, d_model, ffn_dim, n_scales, dropout):
        super().__init__()
        self.scale_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))
            for _ in range(n_scales)
        ])
        self.final_mlp = nn.Sequential(nn.Linear(d_model * n_scales, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, 1))

    def forward(self, dec_outputs):
        pooled = []
        for i, f in enumerate(dec_outputs):
            v = f.mean(dim=(1, 2))
            v = self.scale_mlps[i](v)
            pooled.append(v)
        cat = torch.cat(pooled, dim=-1)
        out = self.final_mlp(cat)
        return out.view(-1)


class STARModelFull(nn.Module):
    def __init__(self, window, n_sensors, d_model, nhead, num_scales,
                 ffn_dim=256, patch_size=4, dropout=0.25,
                 encoder_layers_per_scale=3, decoder_layers_per_scale=2,
                 pos_learnable=True, is_teacher=False, target_noise_std=0.05,
                 num_regimes=6, num_faults=2, future_len=20):
        super().__init__()
        self.is_teacher = is_teacher
        self.num_scales = num_scales

        self.patch_embed = PatchEmbedDimWise(window, n_sensors, patch_size, d_model, pos_learnable)

        if self.is_teacher:
            self.future_proj = nn.Linear(future_len * n_sensors, d_model)
            self.regime_emb = nn.Embedding(num_regimes, d_model)
            self.fault_emb = nn.Embedding(num_faults, d_model)
            self.teacher_noise_std = target_noise_std

            nn.init.normal_(self.regime_emb.weight, std=0.02)
            nn.init.normal_(self.fault_emb.weight, std=0.02)
            nn.init.zeros_(self.future_proj.weight)
            nn.init.zeros_(self.future_proj.bias)
        else:
            self.projectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model)
                ) for _ in range(num_scales)
            ])

        self.encoder = STAREncoder(num_scales, d_model, nhead, ffn_dim, dropout, encoder_layers_per_scale)
        self.decoder = STARDecoder(num_scales, d_model, nhead, ffn_dim, dropout, decoder_layers_per_scale)
        self.pred_head = PredictionHead(d_model, ffn_dim, num_scales, dropout)

    def forward(self, x, x_future=None, y_meta=None, return_latents=False):
        emb = self.patch_embed(x)
        B, S, T, d = emb.shape

        if self.is_teacher and x_future is not None and y_meta is not None:
            fut_flat = x_future.view(B, -1)

            if self.training:
                noise = torch.randn_like(fut_flat) * self.teacher_noise_std
                fut_flat = fut_flat + noise

            fut_bias = self.future_proj(fut_flat).unsqueeze(1).unsqueeze(1)

            regime_id = y_meta[:, 1].long()
            fault_id = y_meta[:, 2].long()

            r_bias = self.regime_emb(regime_id).unsqueeze(1).unsqueeze(1)
            f_bias = self.fault_emb(fault_id).unsqueeze(1).unsqueeze(1)

            emb = emb + fut_bias + r_bias + f_bias

        enc_feats = self.encoder(emb)

        dec_outs = []
        dec_input = None

        for i in range(len(enc_feats)):
            enc_feat = enc_feats[i]
            B_e, S_e, T_e, d_e = enc_feat.shape

            if dec_input is None:
                pe = positional_encoding(T_e, d_e, device=x.device)
                dec_input = pe.unsqueeze(0).unsqueeze(0).repeat(B_e, S_e, 1, 1)

            cur = self.decoder(dec_input, enc_feat, self.decoder.blocks[i])
            dec_outs.append(cur)
            dec_input = cur

        out = self.pred_head(dec_outs)

        if return_latents:
            if self.is_teacher:
                return out, enc_feats
            else:
                projected_feats = []
                for i, feat in enumerate(enc_feats):
                    projected_feats.append(self.projectors[i](feat))
                return out, projected_feats

        return out
