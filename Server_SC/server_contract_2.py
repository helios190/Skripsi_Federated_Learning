from typing import Dict, Optional, Tuple
import flwr as fl
from utils import model, getMnistDataSet, plotServerData
from web3 import Web3
import json
from utils import build_resnet11, build_resnet17, build_resnet56

# Connect to local Ethereum node
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# Smart contract address and ABI
contract_address = "0x4B7046Bea0EeFD2157582b3268eD7833802aa5B2"
contract_address = w3.toChecksumAddress(contract_address)

# Load contract ABI

with open('/Users/bintangrestubawono/Documents/fl7/FL-Demo-With-Flower/contracts/artifacts/contracts/FederatedLearning.sol/FederatedLearning.json') as f:
    contract_abi = json.load(f)["abi"]

# Initialize contract
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Compile the model
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

results_list = []

def get_eval_fn(model):
    x_train, y_train, x_test, y_test = getMnistDataSet()

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("After round {}, Global accuracy = {} ".format(server_round, accuracy))
        results = {"round": server_round, "loss": loss, "accuracy": accuracy}
        results_list.append(results)

        # Check if the "client_ids" key is present in the config
        if "client_ids" in config:
            for client in config["client_ids"]:
                try:
                    tx_hash = contract.functions.registerClient().transact({'from': client})
                    w3.eth.waitForTransactionReceipt(tx_hash)
                except Exception as e:
                    print(f"Error registering client {client}: {e}")

        return loss, {"accuracy": accuracy}

    return evaluate

strategy = fl.server.strategy.FedAvg(evaluate_fn=get_eval_fn(model))

fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=21),
    strategy=strategy
)