import torch
import torch.nn as nn

class StarBaseline(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_layers,
        dropout,
        patch_size,
        seq_len,
        n_sensors,
        pos_learnable=True
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_sensors, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        if pos_learnable:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, seq_len, d_model)
            )
        else:
            self.pos_embed = None

        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x.squeeze(-1)
