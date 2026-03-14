"""
Model building utilities for constructing backbone and head layers from YAML configs.
"""

import torch.nn as nn

from coda.models.modules import Conv, Focus, FiLMConv, Bottleneck, Concat
from coda.utils.general import make_divisible


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
                nn.init.constant_(param[m.hidden_size:m.hidden_size*2], 1)  # fg bias
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    if isinstance(m, nn.ELU):
        m.inplace = True


def parse_model(d, ch):  # model_dict, input_channels
    anchors, nc = d['anchors'], d['nc']
    activation = eval(d.get('activation', 'nn.ELU'))
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * 5  # number of outputs = anchors * 5 (bbox + objectness)
    groupnorm = d.get('groupnorm', False)
    zdim = d['encoder']['params']['zdim']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, m, args) in enumerate(d['backbone'] + d['head']):  # from, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        if m in [Conv, Focus, FiLMConv, Bottleneck]:
            c1, c2 = ch[f], args[0]

            c2 = make_divisible(c2, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]

        elif m in [nn.BatchNorm2d, nn.GroupNorm]:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        else:
            c2 = ch[f]

        if m in [Conv, Focus, Bottleneck]:
            m_ = m(*args, groupnorm=groupnorm, activation=activation)  # module
        elif m == FiLMConv:
            m_ = m(*args, zdim=zdim, groupnorm=groupnorm, activation=activation)  # module
        else:
            m_ = m(*args)  # module

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)

    model = nn.Sequential()

    for i, layer in enumerate(layers):
        model.add_module(f'{layer._get_name()}_{i}', layer)

    return model, sorted(save)
