# -*- coding: utf-8 -*-  
from tf_prototxt import *

def residual(j, num_output, num, char):
    stride = 1
    if num == 2 and j == 0:
        stride = 1
    elif j == 0:
        stride = 2
    char = str(num) + char
    last_num_output = num_output * 4
    x = cc.top()
    if j == 0:
        L.conv("_branch1", num_output=last_num_output, kernel_size=1, stride=stride, bias_term=False)
        with cc.noscope():
            L.BatchNorm("bn%s_branch1" % char)
            L.scale("scale%s_branch1" % char)
        shortcut = cc.top()
    else:
        shortcut = x

    cc.top(x)
    L.conv("_branch2a", kernel_size=1, num_output=num_output, stride=stride, bias_term=False)
    with cc.noscope():
        L.BatchNorm("bn%s_branch2a" % char)
        L.scale("scale%s_branch2a" % char)
    L.relu("_branch2a_relu")

    L.conv("_branch2b", num_output=num_output, kernel_size=3, pad=1)
    with cc.noscope():
        L.BatchNorm("bn%s_branch2b" % char)
        L.scale("scale%s_branch2b" % char)
    L.relu("_branch2b_relu")

    L.conv("_branch2c", kernel_size=1, num_output=last_num_output)
    with cc.noscope():
        L.BatchNorm("bn%s_branch2c" % char)
        L.scale("scale%s_branch2c" % char)

    L.Eltwise(bottoms=[cc.top(), shortcut])
    L.relu("_relu")


def CNN():
    cc.default_conv_param(kernel_size=1, num_output=64, pad=0, stride=1)
    cc.default_pooling_param(pool=Enum("MAX"), kernel_size=3, stride=2)
    cc.default_inner_product_param(inner_product_param={"num_output":64})
    cc.default_batchnorm_param(batch_norm_param=dict(use_global_stats=True))
    cc.default_scale_param(scale_param=dict(bias_term=True))

    L.conv("conv1", num_output=64, kernel_size=7, pad=3, stride=2)
    L.BatchNorm("bn_conv1")
    L.scale("scale_conv1")
    L.relu("conv1_relu")
    L.pooling("pool1")

    num_output = 64
    for i in range(4):
        num = i + 2
        ran = 3
        if i == 1:
            ran = 4
        elif i == 2:
            ran = 6
        for j in range(ran):
            char = chr(j + 97)
            with cc.scope("res%d%s" % (num, char)):
                residual(j, num_output, num, char)
        num_output *= 2

    L.pooling("pool5", kernel_size=7, stride=1, pool=Enum("AVE"))
    L.inner_product("fc1000", num_output=1000)

def train_ptototxt():
    cc.clean()
    cc.name("ResNet50-train")
    data, label = L.LMDBData(source="train_lmdb", batch_size=10, phase=Enum("TRAIN"))
    data, label = L.LMDBData(source="test_lmdb", batch_size=10, phase=Enum("TEST"))
    CNN()
    L.SoftmaxWithLoss(label, phase=Enum("TRAIN"))
    L.Accuracy(label, phase=Enum("TEST"))
    return cc.prototxt()

def deploy_prototxt():
    cc.clean()
    cc.name("ResNet50-test")
    L.InputLayer(dims=[1, 3, 224, 224])
    CNN()
    L.softmax()
    return cc.prototxt()

def solver():
    s = Solver()
    s.base_lr = .11
    return str(s)

print(solver())