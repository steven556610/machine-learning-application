import torch
import torch.nn as nn
import logging

from src.log_module import Logger

class SmallTransformerClassifier(Logger):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_dim, log_level=logging.INFO):
        """
        Initializes a small Transformer-based text classifier.

        This model uses a Transformer Encoder to process an input sequence of word vectors
        and outputs a single classification probability.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the word embeddings.
            nhead (int): The number of heads in the multi-head attention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
            output_dim (int): The number of output classes (e.g., 1 for binary classification).
            log_level (int): The logging level for this class.
        """
        super().__init__(log_level=log_level)
        self.model = self._create_model(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim
        )
        self.logger.info("Transformer classifier model defined.")

    def _create_model(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_dim):
        """
        A helper function to build the model's architecture.
        """
        # The embedding layer is crucial for mapping vocabulary indices to vectors.
        # We will initialize this with pre-trained Word2Vec vectors later.
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # A simple positional encoding to give the model information about word order.
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # The core Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # A global average pooling layer to collapse the sequence of vectors into a single vector
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # A final classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
            nn.Sigmoid() # Use Sigmoid for binary classification
        )

        return self

    def forward(self, x, src_mask=None, src_padding_mask=None):
        """
        Forward pass of the model.
        Args:
            x (Tensor): The input tensor of token indices.
            src_mask (Tensor, optional): A mask to prevent attention to future tokens.
            src_padding_mask (Tensor, optional): A mask to ignore padded tokens.
        """
        # x is a tensor of shape (batch_size, sequence_length) with token indices.
        # We look up their vectors using the embedding layer.
        x = self.embedding(x) * math.sqrt(self.model.d_model)
        x = self.pos_encoder(x)
        
        # Pass the word vectors through the Transformer encoder.
        output = self.transformer_encoder(x, src_mask, src_padding_mask)
        
        # Global average pooling to get a single vector per document.
        output = output.permute(0, 2, 1) # Reshape for AdaptiveAvgPool1d
        output = self.avg_pool(output).squeeze(2)
        
        # Final classification
        return self.classifier(output)

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding module for Transformer models.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, sequence_length, embedding_dim)
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)