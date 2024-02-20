import math
from torch import nn

import torch

from model.EncoderALLFlower import encoderALL
# from model.EncoderALL import encoderALL
from model.UformerY import BasicUformerLayer, InputProj


def ConverBSHW(x):
    B, L, C = x.shape
    # import pdb;pdb.set_trace()
    H = int(math.sqrt(L))
    W = int(math.sqrt(L))
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x


# Upsample Block
class upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x,convert=False):
        if not convert:
            B, L, C = x.shape
            H = int(math.sqrt(L))
            W = int(math.sqrt(L))
            x = x.transpose(1, 2).contiguous().view(B, C, H, W)
            out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        else:
            out = self.deconv(x)  # B  C H W
        return out

class DecoderFlower(nn.Module):
    def __init__(self,  img_size=256, dd_in=1,
                 embed_dim=16, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff',
                 shift_flag=False, modulator=False,cross_modulator=False, **kwargs):
        super(DecoderFlower, self).__init__()
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in
        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]
        # Bottleneck


        self.decoderlayer_f = BasicUformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)


        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)

        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim ,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)

        self.sig1 = nn.Sigmoid()
        self.output_proj = OutputProj(in_channel= embed_dim, out_channel=dd_in, kernel_size=3, stride=1)


        # 以下是y的信息

        self.upsample_01 = upsample(embed_dim * 16, embed_dim * 8)

        self.decoderlayer_01 = BasicUformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_11 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_11 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_21 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_21 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)
        self.upsample_31 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_31 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator, cross_modulator=cross_modulator)

        self.sig11 = nn.Sigmoid()
        self.output_proj1 = OutputProj(in_channel=embed_dim, out_channel=dd_in, kernel_size=3, stride=1)


    # feature is list[32 *128*128 64*64*64 128*32*32 256*16*16     ]
    def forward(self, feature,mask=None):
        for ii in range(4):
            feature[ii] = feature[ii].flatten(2).transpose(1, 2).contiguous() # B H*W C



        deconv_f = self.decoderlayer_f(feature[3], mask=mask)


        # Decoder1
        up0 = self.upsample_0(deconv_f)
        deconv0 = torch.cat([up0, feature[2]], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, feature[1]], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, feature[0]], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        up3 = self.upsample_3(deconv2)

        deconv3 = self.decoderlayer_3(up3, mask=mask)

        y1 = self.output_proj(deconv3)
        y1 = self.sig1(y1)


        # Decoder2
        up01 = self.upsample_01(deconv_f)
        deconv01 = torch.cat([up01, feature[2]], -1)
        deconv01 = self.decoderlayer_01(deconv01, mask=mask)

        up11 = self.upsample_11(deconv01)
        deconv11 = torch.cat([up11, feature[1]], -1)
        deconv11 = self.decoderlayer_11(deconv11, mask=mask)

        up21 = self.upsample_21(deconv11)
        deconv21 = torch.cat([up21, feature[0]], -1)
        deconv21 = self.decoderlayer_2(deconv21, mask=mask)

        up31 = self.upsample_31(deconv21)

        deconv31 = self.decoderlayer_31(up31, mask=mask)

        y11 = self.output_proj1(deconv31)
        y11 = self.sig11(y11)






        return y1,y11

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops


# if __name__ == '__main__':
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '5'
#     h = 256
#     w = 256
#     x = torch.randn(1,2,h,w,4).cuda()
#     img1 = torch.randn(1,3,h,w).cuda()
#
#     nb_of_time_bin = 2
#     netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
#     model = encoderALL(netParams,imageSize=h,channel=3).cuda()
#
#     de = DecoderFlower().cuda()
#
#
#
#     import time
#     start = time.time()
#     y  =   model(x,x,img1,torch.as_tensor(2).cuda().resize(1,),torch.as_tensor(2).resize(1,).cuda())
#     de(y)


