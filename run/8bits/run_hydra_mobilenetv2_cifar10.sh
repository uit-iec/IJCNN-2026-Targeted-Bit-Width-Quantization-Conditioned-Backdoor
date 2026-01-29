python hydra_backdoor_w_lossfn.py -m \
    dataset=cifar10\
    model=mobilenetv2\
    model.trained=pretrained\
    params.batch_size=64\
    params.epoch.pretrained=50\
    params.lr.pretrained=1e-5\
    hydra.mode=RUN