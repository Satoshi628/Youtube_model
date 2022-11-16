python3 main.py hydra.run.dir=./outputs/Resnet50 \
    train_conf.gpu=0 \
    train_conf.epoch=300 \
    train_conf.pretrained=True \
    train_conf.batch_size=256 \
    train_conf.test_size=256