from torch import nn
from cross_attn import CrossAttentionBlock
import timm

class SwinCA(nn.Module):
    def __init__(self):
        super().__init__()
        swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        self.swin_layer1_1 = nn.Sequential(
            swin.patch_embed, swin.pos_drop, swin.layers[0]
        )
        self.swin_layer1_2 = swin.layers[1]
        self.swin_layer1_3 = swin.layers[2]
        self.swin_layer1_4 = nn.Sequential(swin.layers[3], swin.norm)
        self.swin_avgpool1 = swin.avgpool

        swin2 = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        self.swin_layer2_1 = nn.Sequential(
            swin2.patch_embed, swin2.pos_drop, swin2.layers[0]
        )
        self.swin_layer2_2 = swin2.layers[1]
        self.swin_layer2_3 = swin2.layers[2]
        self.swin_layer2_4 = nn.Sequential(swin2.layers[3], swin2.norm)
        self.swin_avgpool2 = swin2.avgpool

        self.cross_attn1 = CrossAttentionBlock(dim=192, num_heads=16)
        self.cross_attn2 = CrossAttentionBlock(dim=384, num_heads=16)
        self.cross_attn3 = CrossAttentionBlock(dim=768, num_heads=16)

        self.final_linear = nn.Linear(768+768, 128)
        self.act = nn.GELU()
        self.output = nn.Linear(128, 1)
        self.output_act = nn.ReLU()


    def forward(self, x, y):
        x = self.swin_layer1_1(x)
        y = self.swin_layer2_1(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn1(z)
        x, y = torch.split(z, [784, 784], dim=1)

        x = self.swin_layer1_2(x)
        y = self.swin_layer2_2(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn2(z)
        x, y = torch.split(z, [196, 196], dim=1)

        x = self.swin_layer1_3(x)
        y = self.swin_layer2_3(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn3(z)
        x, y = torch.split(z, [49, 49], dim=1)

        x = self.swin_layer1_4(x)
        x = self.swin_avgpool1(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        y = self.swin_layer2_4(y)
        y = self.swin_avgpool2(y.transpose(1, 2))
        y = torch.flatten(y, 1)

        z = torch.cat([x, y], dim=1)
        z = self.final_linear(z)
        output = self.output(self.act(z))
        output = self.output_act(output)

        return output