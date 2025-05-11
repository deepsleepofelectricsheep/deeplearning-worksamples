import torch
from torch import hub
from torch import nn


class ViTBackbone(nn.Module):
    """A ViT backbone for sparse classification that returns the 
    class token embedding, which can be used by a model head for
    sparse prediction tasks. 

    Attributes:
        model:
            Instance of the DinoVisionTransformer class, which
            takes tensors of size (B, 3, H, W) and returns a 
            tensor of shape (B, embed_dim), where embed_dim
            depends on the model variant. 
    """

    def __init__(
        self,
        repo: str = "facebookresearch/dinov2",
        model: str = "dinov2_vits14",
        freeze_backbone: bool = True
    ) -> None:
        """Initializes instance of the VitBackbone, based on the 
        provided arguments. 

        Args:
            repo: 
                Github repo that contains the pretrained model.
            model:
                The model name, as defined in the repo's hubconf.py 
                file.
            freeze_backbone:
                Boolean flag that determines whether to freeze model
                parameters. 
        """
        super().__init__()
        self.model = hub.load(repo, model)
        if freeze_backbone == True:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Forward pass of ViT model.
        
        Args:
            xb: Input batch of shape (B, 3, H, W).

        Returns:
            Class token embedding of shape (B, self.model.embed_dim).
        """
        return self.model(xb)
    

class ClassificationHead(nn.Module):
    """A simple, MLP-based model for classification.
    
    Attributes:
        model: PyTorch module for classification.
    """

    def __init__(
        self,
        input_size: int = 384,
        hidden_size: int = 512, 
        output_size: int = 10
    ) -> None:
        """Initializes instance of MLP classification model.
        
        Args:
            input_size: Size of model input.
            hidden_size: Size of hidden layer.
            output_size: Size of output. 
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        """Basic forward pass of model.
        
        Args: 
            xb: Input batch of shape (B, I)

        Returns:
            Logits of shape (B, O)
        """
        return self.model(xb)
