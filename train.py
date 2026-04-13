from run import DGR_run

DGR_run(
    model_name='DGR',
    dataset_name='mosi',#更换数据集
    is_tune=False,
    seeds=[68],
    model_save_dir="./pt",
    res_save_dir="./result",
    log_dir="./log",
    mode='train',
    is_training=True
)
