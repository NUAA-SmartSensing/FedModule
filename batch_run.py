import yaml
import subprocess
import copy

# 需要尝试的lr取值
lr_list = [0.1, 0.01, 0.05]

# 读取yaml
with open('config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

for idx, lr in enumerate(lr_list, 1):
    config = copy.deepcopy(base_config)
    # 修改lr参数
    config['client']['optimizer']['params']['lr'] = lr
    # 修改wandb.name和experiment字段
    config['wandb']['name'] = f"lr_{lr}"
    config['global']['experiment'] = f"FedAsync/lr_{lr}"
    # 保存yaml
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    # 运行主程序
    subprocess.run(['python3', 'src/fl/main.py', 'config.yaml'])

