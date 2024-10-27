"""flower-dp: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flower_dp.task import Net, get_weights

from typing import List, Tuple

import wandb


use_dp = True
project = "FlowerDP"

global_run = wandb.init(
    project=project,
    config={
        "num_clients": 20,
        "num_rounds": 30,
        "local_epochs": 2,
        "fraction_fit": 0.5,
        "target_delta": 1e-5,
        "max_grad_norm": 1.0,
        "use_dp": use_dp,
    },
)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    losses = [num_examples * m["test_loss"] for num_examples, m in metrics]
    accuracy = sum(accuracies) / sum(examples)
    loss = sum(losses) / sum(examples)

    global_run.log({"accuracy": accuracy, "test_loss": loss})
    return {"accuracy": accuracy}


def epsilon_metric(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    loss = sum(losses) / sum(examples)

    if use_dp:
        epsilons = [num_examples * m["epsilon"] for num_examples, m in metrics]
        epsilon = sum(epsilons) / sum(examples)

        global_run.log({"epsilon": epsilon, "train_loss": loss})
        return {"epsilon": epsilon}

    else:
        global_run.log({"train_loss": loss})
        return {}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        fit_metrics_aggregation_fn=epsilon_metric,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
