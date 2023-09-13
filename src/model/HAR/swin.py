import math
import torch
import torch.nn as nn

'''Swin Transformer'''


class ShiftWindowAttentionBlock(nn.Module):
    def __init__(self, patch_num, input_dim=512, head_num=4, att_size=64, window_size=4, shift=False):
        super().__init__()
        '''
            patch_num: 输入的patch数, 即数据的倒数第二维
            input_dim: 输入维度, 即embedding维度
            head_num: 多头自注意力
            att_size: QKV矩阵维度
            window_size: 一个窗口包含多少patchs
            shift: 是否窗口偏移
        '''
        self.patch_num = patch_num
        self.head_num = head_num
        self.att_size = att_size
        self.window_size = window_size
        self.window_num = self.patch_num // self.window_size  # window_num
        self.shift_size = window_size // 2 if shift else 0
        self.query = nn.Linear(input_dim, head_num * att_size)
        self.key = nn.Linear(input_dim, head_num * att_size)
        self.value = nn.Linear(input_dim, head_num * att_size)

        # 判断是否使用窗口移位，如果使用窗口移位，会使用mask windows
        if self.shift_size:
            img_mask = torch.zeros((1, 1, self.patch_num, 1))  # [1, 1, patch_num, 1]
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for w in w_slices:
                img_mask[:, :, w, :] = cnt
                cnt += 1
            mask_windows = img_mask.reshape(1, 1, self.window_num, self.window_size,
                                            1)  # [1, 1, window_num, window_size, 1]
            mask_windows = mask_windows.view(-1, self.window_size)  # [window_num, window_size]
            self.attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
                2)  # [window_num, window_size, window_size]
            self.attn_mask = self.attn_mask.masked_fill(self.attn_mask != 0, float(-100.0)).masked_fill(
                self.attn_mask == 0, float(0.0))  # [window_num, window_size, window_size]

        # 多头自注意力后从att_size恢复成input_dim
        self.att_mlp = nn.Sequential(
            nn.Linear(head_num * att_size, input_dim),
            nn.LayerNorm(input_dim)
        )

        # 多头自注意力后后进行前向全连接
        self.forward_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

        # 每两个ShiftWindowAttentionBlock（即每个swin_transformer_block）进行一次1/2降采样并恢复维度
        if self.shift_size:
            self.downsample_mlp = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.LayerNorm(input_dim)
            )

    def forward(self, x):
        '''
            x: [batch, modal_leng, patch_num, input_dim]
        '''
        # patch_num补成能够被window_size整除
        if x.size(-2) % self.window_size:
            x = nn.ZeroPad2d((0, 0, 0, self.window_size - x.size(-2) % self.window_size))(x)

        batch, modal_leng, patch_num, input_dim = x.size()
        short_cut = x  # resdual

        # 窗口偏移
        if self.shift_size:
            x = torch.roll(x, shifts=-self.shift_size,
                           dims=2)  # 只在 patch_num 上 roll   [batch, modal_leng, patch_num, input_dim]

        # 窗口化 
        window_num = patch_num // self.window_size
        window_x = x.reshape(batch, modal_leng, window_num, self.window_size,
                             input_dim)  # [batch, modal_leng, window_num, window_size, input_dim]

        # 基于窗口的多头自注意力
        q = self.query(window_x).reshape(batch, modal_leng, window_num, self.window_size, self.head_num,
                                         self.att_size).permute(0, 1, 2, 4, 3,
                                                                5)  # [batch, modal_leng, window_num, head_num, window_size, att_size]
        k = self.key(window_x).reshape(batch, modal_leng, window_num, self.window_size, self.head_num,
                                       self.att_size).permute(0, 1, 2, 4, 5,
                                                              3)  # [batch, modal_leng, window_num, head_num, att_size, window_size]
        v = self.value(window_x).reshape(batch, modal_leng, window_num, self.window_size, self.head_num,
                                         self.att_size).permute(0, 1, 2, 4, 3,
                                                                5)  # [batch, modal_leng, window_num, head_num, window_size, att_size]
        attn = torch.matmul(q,
                            k) / self.att_size ** 0.5  # [batch, modal_leng, window_num, head_num, window_size, window_size]
        if self.shift_size:  # 判断是否使用窗口移位，如果使用窗口移位，会使用mask windows
            attn = attn + self.attn_mask.unsqueeze(1).unsqueeze(0).unsqueeze(0).to(
                attn.device)  # [batch, modal_leng, window_num, head_num, window_size, window_size] + [1, 1, window_num, 1, window_size, window_size]
        z = torch.matmul(nn.Softmax(dim=-1)(attn),
                         v)  # [batch, modal_leng, window_num, head_num, window_size, att_size]

        # 反窗口化 
        z = z.permute(0, 1, 2, 4, 3, 5).reshape(batch, modal_leng, window_num * self.window_size,
                                                self.head_num * self.att_size)  # [batch, modal_leng, window_num*window_size, head_num * att_size]
        z = self.att_mlp(z)  # [batch, modal_leng, patch_num, input_dim]

        # 窗口回移
        if self.shift_size:
            z = torch.roll(z, shifts=self.shift_size,
                           dims=2)  # 只在 patch_num 上 roll   [batch, modal_leng, patch_num, input_dim]

        # resdual
        z = nn.ReLU()(short_cut + z)  # [batch, modal_leng, patch_num, input_dim]

        # mlp + resdual
        out = nn.ReLU()(z + self.forward_mlp(z))  # [batch, modal_leng, patch_num, input_dim]

        # 遇到一次窗口偏移就进行 1/2 降采样
        if self.shift_size:
            out = self.patch_merging(out)  # [batch, modal_leng, patch_num/2, input_dim]
        return out

    def patch_merging(self, x):
        '''
            用于进行 1/2 降采样
            x.shape: [batch, modal_leng, patch_num, input_dim]
        '''
        batch, modal_leng, patch_num, input_dim = x.shape
        if patch_num % 2:  # patch_num补成偶数方便1/2降采样
            x = nn.ZeroPad2d((0, 0, 0, 1))(x)
        x0 = x[:, :, 0::2, :]  # [batch, modal_leng, patch_num / 2, input_dim]
        x1 = x[:, :, 1::2, :]  # # [batch, modal_leng, patch_num / 2, input_dim]
        x = torch.cat([x0, x1], dim=-1)  # [batch, modal_leng, patch_num / 2, input_dim * 2]
        x = nn.ReLU()(self.downsample_mlp(x))  # [batch, modal_leng, patch_num / 2, input_dim]
        return x


class SwinTransformer(nn.Module):
    def __init__(self, train_shape, category, embedding_dim=256, patch_size=4, head_num=4, att_size=64, window_size=8):
        super().__init__()
        '''
            train_shape: 总体训练样本的shape
            category: 类别数
            embedding_dim: embedding 维度
            patch_size: 一个patch长度
            head_num: 多头自注意力
            att_size: QKV矩阵维度
            window_size: 一个窗口包含多少patchs
        '''
        # cut patch
        # 对于传感窗口数据来讲，在每个单独的模态轴上对时序轴进行patch切分
        # 例如 uci-har 数据集窗口尺寸为 [128, 9]，一个patch包含4个数据，那么每个模态轴上的patch_num为32, 总patch数为 32 * 9
        self.series_leng = train_shape[-2]
        self.modal_leng = train_shape[-1]
        self.patch_num = self.series_leng // patch_size

        self.patch_conv = nn.Conv2d(
            in_channels=1,
            out_channels=embedding_dim,
            kernel_size=(patch_size, 1),
            stride=(patch_size, 1),
            padding=0
        )

        # 位置信息
        self.position_embedding = nn.Parameter(torch.zeros(1, self.modal_leng, self.patch_num, embedding_dim))

        # patch_num维度降采样一次后的计算方式
        swin_transformer_block1_input_patch_num = math.ceil(self.patch_num / window_size) * window_size
        swin_transformer_block2_input_patch_num = math.ceil(
            math.ceil(swin_transformer_block1_input_patch_num / 2) / window_size) * window_size
        swin_transformer_block3_input_patch_num = math.ceil(
            math.ceil(swin_transformer_block2_input_patch_num / 2) / window_size) * window_size

        # Shift_Window_Attention_Layer
        # 共3个swin_transformer_block，每个swin_transformer_block对时序维降采样1/2，共降采样1/8
        self.swa = nn.Sequential(
            # swin_transformer_block 1
            nn.Sequential(
                ShiftWindowAttentionBlock(patch_num=swin_transformer_block1_input_patch_num, input_dim=embedding_dim,
                                          head_num=head_num, att_size=att_size, window_size=window_size, shift=False),
                ShiftWindowAttentionBlock(patch_num=swin_transformer_block1_input_patch_num, input_dim=embedding_dim,
                                          head_num=head_num, att_size=att_size, window_size=window_size, shift=True)
            ),
            # swin_transformer_block 2
            nn.Sequential(
                ShiftWindowAttentionBlock(patch_num=swin_transformer_block2_input_patch_num, input_dim=embedding_dim,
                                          head_num=head_num, att_size=att_size, window_size=window_size, shift=False),
                ShiftWindowAttentionBlock(patch_num=swin_transformer_block2_input_patch_num, input_dim=embedding_dim,
                                          head_num=head_num, att_size=att_size, window_size=window_size, shift=True)
            ),
            # swin_transformer_block 3
            nn.Sequential(
                ShiftWindowAttentionBlock(patch_num=swin_transformer_block3_input_patch_num, input_dim=embedding_dim,
                                          head_num=head_num, att_size=att_size, window_size=window_size, shift=False),
                ShiftWindowAttentionBlock(patch_num=swin_transformer_block3_input_patch_num, input_dim=embedding_dim,
                                          head_num=head_num, att_size=att_size, window_size=window_size, shift=True)
            )
        )

        # classification tower
        self.dense_tower = nn.Sequential(
            nn.Linear(self.modal_leng * math.ceil(swin_transformer_block3_input_patch_num / 2) * embedding_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, category)
        )

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x = self.patch_conv(x)  # [batch, embedding_dim, patch_num, modal_leng]
        x = self.position_embedding + x.permute(0, 3, 2, 1)  # [batch, modal_leng, patch_num, embedding_dim]
        # [batch, modal_leng, patch_num, input_dim]
        #  
        # -> [batch, modal_leng, patch_num, input_dim] 
        # -> [batch, modal_leng, patch_num/2, input_dim] 
        #
        # -> [batch, modal_leng, patch_num/2, input_dim]
        # -> [batch, modal_leng, patch_num/4, input_dim]
        #
        # -> [batch, modal_leng, patch_num/4, input_dim]
        # -> [batch, modal_leng, patch_num/8, input_dim]
        x = self.swa(x)
        x = nn.Flatten()(x)
        x = self.dense_tower(x)
        return x
