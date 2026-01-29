python hydra_backdoor_w_lossfn.py -m \
    dataset=cifar100\
    model=vgg16\
    model.trained=pretrained\
    params.scheduler=cosine\
    params.batch_size=256\
    params.epoch.pretrained=200\
    attack.const1=1\
    attack.const2=0.5\
    params.lr.pretrained=0.00008\
    hydra.mode=RUN