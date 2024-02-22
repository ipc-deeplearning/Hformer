

#更新D和dowm的部分
def set_D_down(module):
    for name, value in module.named_parameters():
        # 设置不更新D
        if name.startswith("decoderlayer_12") or name.startswith("decoderlayer_13") or name.startswith(
                "decoderlayer_10") or name.startswith("decoderlayer_11") \
                or name.startswith("upsample_10") or name.startswith("upsample_11") or name.startswith(
            "upsample_12") or name.startswith("upsample_13") \
                or name.startswith("output_proj2"):
            value.requires_grad = False

        else:
            value.requires_grad = True
            # setup optimizer
    params1 = filter(lambda p: p.requires_grad, module.parameters())
    return  params1

#更新S和dowm的部分
def set_S_down(module):
    for name, value in module.named_parameters():
        # 判断是否为S分支,S就不更新
        if name.startswith("decoderlayer_0.") or name.startswith("decoderlayer_1.") or name.startswith(
                "decoderlayer_2.") or name.startswith("decoderlayer_3.") \
                or name.startswith("upsample_0.") or name.startswith("upsample_1.") or name.startswith(
            "upsample_2.") or name.startswith("upsample_3.") \
                or name.startswith("output_proj."):
            value.requires_grad = False
        else:
            value.requires_grad = True

            # setup optimizer
    params2 = filter(lambda p: p.requires_grad, module.parameters())
    return params2


def set_S_D(module):
    for name, value in module.named_parameters():
        if name.startswith("decoderlayer_0.") or name.startswith("decoderlayer_1.") or name.startswith(
                "decoderlayer_2.") or name.startswith("decoderlayer_3.") \
                or name.startswith("upsample_0.") or name.startswith("upsample_1.") or name.startswith(
            "upsample_2.") or name.startswith("upsample_3.") \
                or name.startswith("output_proj."):
            value.requires_grad = True
        elif name.startswith("decoderlayer_12") or name.startswith("decoderlayer_13") or name.startswith(
                "decoderlayer_10") or name.startswith("decoderlayer_11") \
                or name.startswith("upsample_10") or name.startswith("upsample_11") or name.startswith(
            "upsample_12") or name.startswith("upsample_13") \
                or name.startswith("output_proj2"):
            value.requires_grad = True
        else:
            value.requires_grad = False
            # setup optimizer
    params3 = filter(lambda p: p.requires_grad, module.parameters())
    return params3

def set_S(module):
    for name, value in module.named_parameters():
        if name.startswith("decoderlayer_0.") or name.startswith("decoderlayer_1.") or name.startswith(
                "decoderlayer_2.") or name.startswith("decoderlayer_3.") \
                or name.startswith("upsample_0.") or name.startswith("upsample_1.") or name.startswith(
            "upsample_2.") or name.startswith("upsample_3.") \
                or name.startswith("output_proj."):
            value.requires_grad = True
        else:
            value.requires_grad = False
            # setup optimizer
    params3 = filter(lambda p: p.requires_grad, module.parameters())
    return params3

def set_D(module):
    for name, value in module.named_parameters():
        if name.startswith("decoderlayer_12") or name.startswith("decoderlayer_13") or name.startswith(
                "decoderlayer_10") or name.startswith("decoderlayer_11") \
                or name.startswith("upsample_10") or name.startswith("upsample_11") or name.startswith(
            "upsample_12") or name.startswith("upsample_13") \
                or name.startswith("output_proj2"):
            value.requires_grad = True

        else:
            value.requires_grad = False


            # setup optimizer
    params3 = filter(lambda p: p.requires_grad, module.parameters())
    return params3




#更新S和dowm的部分
def Y_set_S_down(module):
    for name, value in module.named_parameters():
        # 判断是否为S分支,S就不更新
        if name.startswith("upx1") or name.startswith("upx2") or name.startswith(
                "upx3") or name.startswith("upx4") :
            value.requires_grad = False
        else:
            value.requires_grad = True

            # setup optimizer
    params2 = filter(lambda p: p.requires_grad, module.parameters())
    return params2
#更新S和dowm的部分
def Y_set_D(module):
    for name, value in module.named_parameters():
        # 判断是否为S分支,S就不更新
        if name.startswith("upx1") or name.startswith("upx2") or name.startswith(
                "upx3") or name.startswith("upx4") :
            value.requires_grad = True
        else:
            value.requires_grad = False

            # setup optimizer
    params3 = filter(lambda p: p.requires_grad, module.parameters())
    return params3