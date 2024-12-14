import flwr as fl
from utils import model, getData, showDataDist, getMnistDataSet, plotClientData
from web3 import Web3
import json
import os

# Connect to local Ethereum node
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# Smart contract address and ABI
contract_address = "0x4B7046Bea0EeFD2157582b3268eD7833802aa5B2"
contract_address = w3.toChecksumAddress(contract_address)

with open('/Users/bintangrestubawono/Documents/fl7/FL-Demo-With-Flower/contracts/artifacts/contracts/FederatedLearning.sol/FederatedLearning.json') as f:
    contract_abi = json.load(f)["abi"]

contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# MetaMask account details
account = "0xcB767eA2c38D6e79Fc27f0a313F14e916f8c57e2"
account = w3.toChecksumAddress(account)
private_key = "0x8b16b36fef9e6262209710a84af6e9c9887b8b1452721639923e535e645d67eb"

# Check account balance
def check_balance():
    balance_wei = w3.eth.get_balance(account)
    balance_eth = w3.fromWei(balance_wei, 'ether')
    print(f"Account balance: {balance_eth} ETH")
    return balance_eth

balance_eth = check_balance()
if balance_eth < 0.1:
    raise ValueError("Insufficient funds in the account. Please add more Ether to the account.")

# Get the initial balance in tokens
try:
    token_balance = contract.functions.getBalance(account).call()
    print(f"Initial Balance: {token_balance} tokens")
except Exception as e:
    print(f"Error getting initial balance: {e}")

# Register client and receive tokens
nonce = w3.eth.get_transaction_count(account)
tx = contract.functions.registerClient().build_transaction({
    'from': account,
    'nonce': nonce,
    'gas': 2000000,
    'gasPrice': w3.toWei('50', 'gwei')
})

signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
w3.eth.wait_for_transaction_receipt(tx_hash)

# Check balance after registration
token_balance = contract.functions.getBalance(account).call()
print(f"Balance after registration: {token_balance} tokens")

# Compile the model
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Directory to save and load model weights
MODEL_WEIGHTS_DIR = './model_weights'
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)

# Function to save the global model weights
def save_global_model(round_number):
    model.save_weights(os.path.join(MODEL_WEIGHTS_DIR, f'global_model_round_{round_number}.weights.h5'))
    print(f"Global model weights saved for round {round_number}.")

# Function to load the global model weights
def load_global_model(round_number):
    model.load_weights(os.path.join(MODEL_WEIGHTS_DIR, f'global_model_round_{round_number}.weights.h5'))
    print(f"Global model weights loaded for round {round_number}.")

# Creating Client Using Blockchain
x_train, y_train, x_test, y_test = getMnistDataSet()
dist = [4000, 10, 4000, 10, 4000, 10, 4000, 10, 4000, 10]
x_train, y_train = getData(dist, x_train, y_train)
showDataDist(y_train)

results_list = []

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        raise Exception("Not implemented")

    def get_parameters(self, config):
        # Return the current local model parameters
        return model.get_weights()

    def fit(self, parameters, config):
        # Set the global model parameters
        self.model.set_weights(parameters)

        # Save global model after setting parameters
        save_global_model(config.get("round", 0))  # Save the model after each round

        # Check balance before deducting gas fee for training
        token_balance = contract.functions.getBalance(account).call()
        print(f"Current Balance before training: {token_balance} tokens")
        
        if token_balance < 10:  # Ensure enough balance for the transaction
            raise ValueError(f"Insufficient token balance: {token_balance}. Minimum required is 10 tokens.")

        # Deduct gas fee for training
        nonce = w3.eth.get_transaction_count(account)
        tx = contract.functions.useGas(account, 10).build_transaction({
            'from': account,
            'nonce': nonce,
            'gas': 2000000,
            'gasPrice': w3.toWei('50', 'gwei')
        })

        signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        w3.eth.wait_for_transaction_receipt(tx_hash)

        # Continue with model training after gas deduction
        history = self.model.fit(
            self.x_train,
            self.y_train,
            32,
            3,
            validation_data=(self.x_test, self.y_test),
            verbose=0
        )

        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        print("Local Training Metrics on client 1: {}".format(results))
        results_list.append(results)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        # Set the global model parameters for evaluation
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, verbose=0)
        num_examples_test = len(self.x_test)
        print("Evaluation accuracy on Client 1 after weight aggregation : ", accuracy) 
        
        return loss, num_examples_test, {"accuracy": accuracy}

# Start Flower client
client = FlwrClient(model, x_train, y_train, x_test, y_test)
fl.client.start_client(
    server_address="localhost:8080", 
    client=client.to_client()
)