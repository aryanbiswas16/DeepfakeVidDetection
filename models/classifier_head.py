import torch.nn as nn

class ClassifierHead(nn.Module):
    """Maps embeddings to binary real/fake classification."""
    
    def __init__(self, in_dim=1024, hidden_dim=512):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, 2)  # real/fake
        )
    
    def forward(self, embeddings):
        return self.mlp(embeddings)
