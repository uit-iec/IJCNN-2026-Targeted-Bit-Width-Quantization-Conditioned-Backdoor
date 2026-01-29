python hydra_backdoor_w_lossfn.py -m \
    dataset=cifar100\
    model=mobilenetv2\
    model.trained=pretrained\
    params.scheduler=cosine\
    params.batch_size=64\
    params.epoch.pretrained=200\
    attack.const1=1\
    attack.const2=1\
    params.lr.pretrained=0.00001\
    hydra.mode=RUN