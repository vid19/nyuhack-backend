import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTM_Stutter(nn.Module):
    """
    SEP-28k ConvLSTM architecture adapted for your multi-label stuttering dataset.
    
    Input shape:
        (B, n_mels, T)
    
    Outputs:
        - logits: (B, n_labels)   # multi-label stuttering labels
    """

    def __init__(
        self,
        n_mels=64,
        n_labels=7,
        conv_channels=64,
        conv_kernel=3,
        lstm_hidden=128,
        lstm_layers=1,
        dropout=0.3
    ):
        super().__init__()

        # ---- Temporal conv block #1 ----
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=(1, conv_kernel),
                padding=(0, conv_kernel // 2)
            ),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(True)
        )

        # ---- Temporal conv block #2 ----
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=(1, conv_kernel),
                padding=(0, conv_kernel // 2)
            ),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(True)
        )

        # ---- Pool over mel dimension (frequency) ----
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # ---- LSTM ----
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False
        )

        # ---- Final classifier head ----
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_labels)   # all labels at once
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, mel):
        """
        mel: (B, n_mels, T)
        """

        if mel.dim() == 3:
            mel = mel.unsqueeze(1)  # -> (B,1,n_mels,T)

        x = self.conv1(mel)
        x = self.conv2(x)

        # (B, C, n_mels, T) -> pool freq -> (B, C, 1, T)
        x = self.freq_pool(x).squeeze(2)   # (B, C, T)

        # -> (B, T, C) for LSTM
        x = x.permute(0, 2, 1)

        # LSTM
        out, (h, _) = self.lstm(x)
        h_last = h[-1]   # (B, H)

        # final classifier
        logits = self.classifier(h_last)

        return logits