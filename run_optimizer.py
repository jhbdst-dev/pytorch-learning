from models.model import SimpleModel
from optimizers.optimizer import create_optimizer


model = SimpleModel()

optimizer = create_optimizer(model, lr=0.01)

print(optimizer.param_groups[0]["lr"])

params = optimizer.param_groups[0]["params"]

print("매개변수 개수 :", len(params))