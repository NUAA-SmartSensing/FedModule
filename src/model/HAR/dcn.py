import torch.nn as nn
import torch

'''Deformable Convolutional Network (DCN): 可变形网络'''


def get_min_value(ks, padsize):
    ks0 = ks[0]
    ks1 = ks[1]
    pd0 = padsize[0]
    pd1 = padsize[1]
    while ks0 - 2 > 1:
        if pd0 > 0:
            ks0 -= 2
            pd0 -= 1
        else:
            break
    while ks1 - 2 > 1:
        if pd1 > 0:
            ks1 -= 2
            pd1 -= 1
        else:
            break
    return (ks0, ks1), (pd0, pd1)


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, bias=None, modulation=False, minit=0, pinit=0):

        super(DeformConv2d, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
        self.padding = (padding, padding) if type(padding) == int else padding
        self.stride = (stride, stride) if type(stride) == int else stride

        self.zero_padding = nn.ZeroPad2d((self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        ks, pd = get_min_value(self.kernel_size, self.padding)
        self.p_conv = nn.Conv2d(inc, 2 * self.kernel_size[0] * self.kernel_size[1], kernel_size=ks, padding=pd,
                                stride=self.stride)
        if pinit:
            # nn.init.normal_(self.p_conv.weight, mean=0, std=pinit)
            nn.init.constant_(self.p_conv.weight, pinit)
        else:
            nn.init.constant_(self.p_conv.weight, 0)

        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, self.kernel_size[0] * self.kernel_size[1], kernel_size=ks, padding=pd,
                                    stride=self.stride)
            nn.init.constant_(self.m_conv.weight, minit)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):  # (256, 1, 128, 9)
        offset = self.p_conv(x)  # (256, 18, 128, 9)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()  # float
        ks = self.kernel_size  # (3, 3)
        N = offset.size(1) // 2  # 9

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)  # (256, 18, 6, 512)
        p = p.contiguous().permute(0, 2, 3, 1)  # (256, 6, 512, 18)

        q_lt = p.detach().floor()  # 向下取整
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size[1] - 1) // 2, (self.kernel_size[1] - 1) // 2 + 1),
            torch.arange(-(self.kernel_size[0] - 1) // 2, (self.kernel_size[0] - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride[0] + 1, self.stride[0]),
            torch.arange(1, w * self.stride[1] + 1, self.stride[1]))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)  # (9, 128, 9)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)  # (1, 18, 1, 1)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)  # (1, 9+9, 128, 9)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat(
            [x_offset[..., s:s + ks[1]].contiguous().view(b, c, h, w * ks[1]) for s in range(0, N, ks[1])], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks[0], w * ks[1])

        return x_offset


class DeformableConvolutionalNetwork(nn.Module):
    def __init__(self, train_shape, category):
        super(DeformableConvolutionalNetwork, self).__init__()
        '''
            train_shape: 总体训练样本的shape【DCN自适应调整卷积形状，不需要固定按模态轴进行条形卷积，因此这里没有用到train_shape来设定adapool与fc】
            category: 类别数
        '''
        self.layer = nn.Sequential(
            DeformConv2d(1, 64, 3, 2, 1, modulation=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            DeformConv2d(64, 128, 3, 2, 1, modulation=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            DeformConv2d(128, 256, 3, 2, 1, modulation=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            DeformConv2d(256, 512, 3, 2, 1, modulation=True),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, 4))
        self.fc = nn.Linear(512 * 4, category)

    def forward(self, x):
        '''
            x.shape: [b, c, series, modal]
        '''
        x = x.permute(0, 1, 3, 2)  # [b, c, modal, series]
        x = self.layer(x)
        x = self.ada_pool(x)  # [b, c, 1, 4]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
