from torch import nn
from .cross_attn import CrossAttentionBlock
import timm
import torch


class ViTCA(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vit = timm.create_model("vit_small_patch16_224_in21k", pretrained=pretrained)
        self.vit_layer1_1 = nn.Sequential(
            vit.patch_embed, vit.pos_drop, vit.blocks[0:3]
        )
        self.vit_layer1_2 = vit.blocks[3:6]
        self.vit_layer1_3 = vit.blocks[6:9]
        self.vit_layer1_4 = nn.Sequential(vit.blocks[9:], vit.norm)
        self.vit_pre_logits1 = vit.pre_logits

        vit2 = timm.create_model("vit_small_patch16_224_in21k", pretrained=pretrained)
        self.vit_layer2_1 = nn.Sequential(
            vit2.patch_embed, vit2.pos_drop, vit2.blocks[0:3]
        )
        self.vit_layer2_2 = vit2.blocks[3:6]
        self.vit_layer2_3 = vit2.blocks[6:9]
        self.vit_layer2_4 = nn.Sequential(vit2.blocks[9:], vit2.norm)
        self.vit_pre_logits2 = vit2.pre_logits

        self.cross_attn1 = CrossAttentionBlock(dim=384, num_heads=16)
        self.cross_attn2 = CrossAttentionBlock(dim=384, num_heads=16)
        self.cross_attn3 = CrossAttentionBlock(dim=384, num_heads=16)

        self.final_linear = nn.Linear(768, 128)
        self.act = nn.GELU()
        self.output = nn.Linear(128, 1)
        self.output_act = nn.ReLU()


    def forward(self, x, y):
        x = self.vit_layer1_1(x)
        y = self.vit_layer2_1(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn1(z)
        x, y = torch.split(z, [196, 196], dim=1)

        x = self.vit_layer1_2(x)
        y = self.vit_layer2_2(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn2(z)
        x, y = torch.split(z, [196, 196], dim=1)

        x = self.vit_layer1_3(x)
        y = self.vit_layer2_3(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn3(z)
        x, y = torch.split(z, [196, 196], dim=1)

        x = self.vit_layer1_4(x)
        x = self.vit_pre_logits1(x[:, 0])
        y = self.vit_layer2_4(y)
        y = self.vit_pre_logits2(y[:, 0])

        z = torch.cat([x, y], dim=1)
        z = self.final_linear(z)
        output = self.output(self.act(z))
        output = self.output_act(output)

        return output

## Improved Version

'''
class ViTCA(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vit = timm.create_model("vit_small_patch16_224_in21k", pretrained=pretrained)
        self.vit_layer1_1 = nn.Sequential(
            vit.patch_embed, vit.pos_drop, vit.blocks[0:3]
        )
        self.vit_layer1_2 = vit.blocks[3:6]
        self.vit_layer1_3 = vit.blocks[6:9]
        self.vit_layer1_4 = vit.blocks[9:]
        self.vit_norm1 = vit.norm
        # self.vit_layer1_4 = nn.Sequential(vit.blocks[9:], vit.norm)
        self.vit_pre_logits1 = vit.pre_logits

        vit2 = timm.create_model("vit_small_patch16_224_in21k", pretrained=pretrained)
        self.vit_layer2_1 = nn.Sequential(
            vit2.patch_embed, vit2.pos_drop, vit2.blocks[0:3]
        )
        self.vit_layer2_2 = vit2.blocks[3:6]
        self.vit_layer2_3 = vit2.blocks[6:9]
        # self.vit_layer2_4 = nn.Sequential(vit2.blocks[9:], vit2.norm)
        self.vit_layer2_4 = vit2.blocks[9:]
        self.vit_norm2 = vit2.norm
        self.vit_pre_logits2 = vit2.pre_logits

        self.cross_attn1 = CrossAttentionBlock(dim=384, num_heads=16)
        self.cross_attn2 = CrossAttentionBlock(dim=384, num_heads=16)
        self.cross_attn3 = CrossAttentionBlock(dim=384, num_heads=16)
        self.cross_attn4 = CrossAttentionBlock(dim=384, num_heads=16)

        self.linear1 = nn.Linear(768, 1024)
        self.act1 = nn.GELU()
        self.linear2 = nn.Linear(1024, 256)
        self.act2 = nn.GELU()
        self.output = nn.Linear(256, 1)
        self.output_act = nn.ReLU()


    def forward(self, x, y):
        x = self.vit_layer1_1(x)
        y = self.vit_layer2_1(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn1(z)
        x, y = torch.split(z, [196, 196], dim=1)

        x = self.vit_layer1_2(x)
        y = self.vit_layer2_2(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn2(z)
        x, y = torch.split(z, [196, 196], dim=1)

        x = self.vit_layer1_3(x)
        y = self.vit_layer2_3(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn3(z)
        x, y = torch.split(z, [196, 196], dim=1)

        x = self.vit_layer1_4(x)
        y = self.vit_layer2_4(y)
        z = torch.cat([x, y], dim=1)
        z = self.cross_attn4(z)
        x, y = torch.split(z, [196, 196], dim=1)

        x = self.vit_norm1(x)
        x = self.vit_pre_logits1(x[:, 0])
        y = self.vit_norm2(y)
        y = self.vit_pre_logits2(y[:, 0])

        z = torch.cat([x, y], dim=1)
        z = self.linear1(z)
        z = self.linear2(self.act1(z))
        z = self.output(self.act2(z))
        output = self.output_act(z)

        return output

'''