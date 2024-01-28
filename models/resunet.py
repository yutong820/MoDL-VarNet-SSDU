import torch
from torch import nn
from torch.nn import functional as F

class ResConvBlock(nn.Module):
     
     def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
          super().__init__()
          
          self.in_chans = in_chans
          self.out_chans = out_chans
          self.drop_prob = drop_prob

          self.layers = nn.Sequential(
               nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
               nn.InstanceNorm2d(out_chans),
               nn.LeakyReLU(negative_slope=0.2, inplace=True),
               nn.Dropout2d(drop_prob),
               nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
               nn.InstanceNorm2d(out_chans),
               nn.LeakyReLU(negative_slope=0.2, inplace=True),
               nn.Dropout2d(drop_prob),
          )
          
          # introduce residual conv (1 * 1 convo to adjust channel number)
          self.residual_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0) if in_chans != out_chans else None
          
     def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_layers(x)
        residual = self.residual_conv(residual) if self.residual_conv is not None else residual
        x += residual  # Element-wise addition of input and output (residual connection)
        return x   
     

class ResUnet(nn.Module):
     def __init__(
          self,
          in_chans: int,
          out_chans: int,
          chans: int = 32,
          num_pool_layers: int = 4,
          drop_prob: float = 0.0,
     ):
          """
          Args:
               in_chans: Number of channels in the input to the U-Net model.
               out_chans: Number of channels in the output to the U-Net model.
               chans: Number of output channels of the first convolution layer.
               num_pool_layers: Number of down-sampling and up-sampling layers.
               drop_prob: Dropout probability.
          """
          super().__init__()

          self.in_chans = in_chans
          self.out_chans = out_chans
          self.chans = chans
          self.num_pool_layers = num_pool_layers
          self.drop_prob = drop_prob

          self.down_sample_layers = nn.ModuleList([ResConvBlock(in_chans, chans, drop_prob)])
          ch = chans
          for _ in range(num_pool_layers - 1):
               self.down_sample_layers.append(ResConvBlock(ch, ch * 2, drop_prob))
               ch *= 2
          self.conv = ResConvBlock(ch, ch * 2, drop_prob)

          self.up_conv = nn.ModuleList()
          self.up_transpose_conv = nn.ModuleList()
          for _ in range(num_pool_layers - 1):
               self.up_transpose_conv.append(ResConvBlock(ch * 2, ch))
               self.up_conv.append(ResConvBlock(ch * 2, ch, drop_prob))
               ch //= 2

          self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
          self.up_conv.append(
               nn.Sequential(
                    ResConvBlock(ch * 2, ch, drop_prob),
                    nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
               )
          )

     def forward(self, image: torch.Tensor) -> torch.Tensor:
          """
          Args:
               image: Input 4D tensor of shape `(N, in_chans, H, W)`.

          Returns:
               Output tensor of shape `(N, out_chans, H, W)`.
          """
          stack = []
          output = image

          # apply down-sampling layers
          for layer in self.down_sample_layers:
               output = layer(output)
               stack.append(output)
               output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

          output = self.conv(output)

          # apply up-sampling layers
          for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
               downsample_layer = stack.pop()
               output = transpose_conv(output)

               # reflect pad on the right/botton if needed to handle odd input dimensions
               padding = [0, 0, 0, 0]
               if output.shape[-1] != downsample_layer.shape[-1]:
                    padding[1] = 1  # padding right
               if output.shape[-2] != downsample_layer.shape[-2]:
                    padding[3] = 1  # padding bottom
               if torch.sum(torch.tensor(padding)) != 0:
                    output = F.pad(output, padding, "reflect")

               output = torch.cat([output, downsample_layer], dim=1)
               output = conv(output)

          return output
     
class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            # 1. upsampling nearest neighbor (nn.Module)
            # 2. conv
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
