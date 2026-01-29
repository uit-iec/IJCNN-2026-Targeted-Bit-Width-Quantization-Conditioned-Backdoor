python hydra_backdoor_w_lossfn.py -m \
    dataset=cifar10\
    model=resnet34\
    model.trained=pretrained\
    params.batch_size=64\
    params.epoch.pretrained=100\
    params.lr.pretrained=1e-5\
    attack.numbit=[4]\
    hydra.mode=RUN