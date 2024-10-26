"""flower-dp: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flower_dp.task import Net, get_weights, load_data, set_weights, test, train

from opacus import PrivacyEngine


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self,
        model,
        trainloader,
        valloader,
        local_epochs,
        target_delta,
        noise_multiplier,
        max_grad_norm,
    ):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, parameters, config):
        model = self.model
        set_weights(self.model, parameters)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        privacy_engine = PrivacyEngine(secure_mode=False)
        (model, optimizer, self.trainloader) = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.trainloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        (loss, epsilon) = train(
            model,
            self.trainloader,
            privacy_engine,
            optimizer,
            self.target_delta,
            self.local_epochs,
            self.device,
        )

        print(f"|| Average training loss: {loss:.5f} || Epsilon: {epsilon:.5f} ||")

        return (get_weights(model), len(self.trainloader.dataset), {})

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return (loss, len(self.valloader.dataset), {"accuracy": accuracy})


def client_fn(context: Context):
    # Load model and data
    model = Net()
    partition_id = context.node_config["partition-id"]

    noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5

    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(
        model,
        trainloader,
        valloader,
        local_epochs,
        context.run_config["target-delta"],
        noise_multiplier,
        context.run_config["max-grad-norm"],
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
