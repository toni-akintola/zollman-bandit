import random
import numpy as np
from emergent.main import AgentModel


def generateInitialData(model: AgentModel):
    initial_data = {
        "a_alpha": random.randint(1, 4),
        "a_beta": random.randint(1, 4),
        "b_alpha": random.randint(1, 4),
        "b_beta": random.randint(1, 4),
    }
    expectations = {
        "a_expectation": initial_data["a_alpha"]
        / (initial_data["a_alpha"] + initial_data["a_beta"]),
        "b_expectation": initial_data["b_alpha"]
        / (initial_data["b_alpha"] + initial_data["b_beta"]),
    }
    initial_data.update(expectations)
    return initial_data


def generateTimestepData(model: AgentModel):
    graph = model.get_graph()

    for _node, node_data in graph.nodes(data=True):
        # agent pulls the "a" bandit arm
        if node_data["a_expectation"] > node_data["b_expectation"]:
            node_data["a_alpha"] += int(
                np.random.binomial(model["num_trials"], model["a_objective"], size=None)
            )
            # TODO
            node_data["a_beta"] += max(0, model["num_trials"] - node_data["a_alpha"])
            node_data["a_expectation"] = node_data["a_alpha"] / (
                node_data["a_alpha"] + node_data["a_beta"]
            )

        # agent pulls the "b" bandit arm
        else:
            node_data["b_alpha"] += int(
                np.random.binomial(model["num_trials"], model["b_objective"], size=None)
            )
            # TODO
            node_data["b_beta"] += max(0, model["num_trials"] - node_data["b_alpha"])
            node_data["b_expectation"] = node_data["b_alpha"] / (
                node_data["b_alpha"] + node_data["b_beta"]
            )

    model.set_graph(graph)


def constructModel() -> AgentModel:
    # Recreating the Zollman Model
    model = AgentModel()
    model.update_parameters({"a_objective": 0.19, "b_objective": 0.71, "num_trials": 5})

    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)

    return model


model = constructModel()
model["num_nodes"] = 20
model.initialize_graph()

timesteps = 20

for _ in range(timesteps):
    model.timestep()

graph = model.get_graph()

for node, node_data in graph.nodes(data=True):
    print(node_data["a_expectation"], node_data["b_expectation"])
