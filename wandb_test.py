import wandb

# 可选：使用离线模式，避免网络相关问题
# import os; os.environ["WANDB_MODE"] = "offline"

# 如果需要认证，可以先 wandb.login()，这里假设已正确登录
run = wandb.init(project="quick-test-project", reinit=True)
wandb.log({"test_metric": 0})
run.finish()
print("Finished minimal run")
