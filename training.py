import torch
import random
import web3
from .server import *
from .client import *
from .datasets import *
from .contract_api import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def fl_training(conf, port):
    ''' FL training process '''

    ''' Initialization of smart contract '''
    print("Initialization of the smart contract...")
    # Connecting the Blockchain
    w3 = web3.Web3(web3.HTTPProvider('http://127.0.0.1:'+str(port),
                   request_kwargs={'timeout': 60 * 10}))

    # Compile and deploy the smart contract
    contract_source_path = r"./contracts/Operations.sol"
    compiled_sol = compile_source_file(contract_source_path)
    contract_id, contract_interface = compiled_sol.popitem()
    address = deploy_contract(w3, contract_interface)
    print("The name of the compiled contract:{0} succeed and deployed at : {1}".format(
        contract_id, address))

    # Initialize the contract and accounts
    ctt = w3.eth.contract(
        address=address,
        abi=contract_interface['abi'])
    vs = []
    deposits = []
    deposit_value = conf["deposit_value"]
    rent_factor = 1

    # sending deposit to the contract account
    print("Initial contract balance:", w3.eth.get_balance(address))
    print("Initial account balance:", w3.eth.get_balance(w3.eth.accounts[1]))
    print("Clients submitting the deposits...")
    for i in range(conf["client_total_num"]):
        v = conf["compute_power"][i]*conf["self_data"] + \
            conf["compute_input"][i]*conf["self_compute"]
        v = int(v*rent_factor)
        vs.append(v)
        deposits.append(deposit_value)
        vs[i] = int(vs[i]*conf["amp_factor"])
        deposits[i] = int(deposits[i]*conf["amp_factor"])
        ret = ctt.functions.Submit(int(conf["compute_power"][i] * conf["self_data"]*conf["amp_factor"]),
                                   int(conf["compute_input"][i] *
                                       conf["self_compute"]*conf["amp_factor"]),
                                   i).transact({"from": w3.eth.accounts[i], 'value': deposits[i]})

    print("Contract balance after receiving the deposit:",
          w3.eth.get_balance(w3.eth.accounts[1]))
    print("Account balance after submitting the deposit:",
          w3.eth.get_balance(w3.eth.accounts[1]))
    print("The initial commitment value:", vs)
    print("The initial deposit:", deposits)

    ''' Initialization of FL '''
    print("\nInitialization of the Federated Learning...")
    # Reading datasets according to configuration
    train_datasets, eval_datasets = get_dataset(
        "./data/", conf["dataset_name"])

    # Initialize server
    server = Server(conf, eval_datasets)

    # Initialize the clients
    clients = []
    for c in range(conf["client_total_num"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    # Initialize the acc and loss lists
    acc_list = []
    lost_list = []

    ''' Training epoch '''
    sns.set()
    print("Training begins...")
    for e in range(conf["global_epoch_num"]):
        # choose k candidates randomly
        candidates = random.sample(clients, conf["client_train_num"])

        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            # Initialize the weights of every part of the network
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            # Train every candidate
            diff = c.local_train(server.global_model)
            for name, params in server.global_model.state_dict().items():
                # Accumulate the difference of every client in every part of the netwrork
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)  # Update the global model

        acc, loss = server.model_eval()  # Evaluation of the updated global model
        acc_list.append(acc)
        lost_list.append(loss)
        print("Epoch %d, acc: %f, loss: %f" % (e+1, acc, loss))

        plt.figure()
        df_loss = pd.DataFrame(
            dict(Epoch=range(1, len(lost_list)+1), Loss=lost_list))
        sns.lineplot(data=df_loss, x='Epoch', y='Loss', errorbar=('ci', 95))
        plt.legend(["Global Loss"], loc='upper right')
        plt.savefig(conf['loss_addr'], dpi=500, bbox_inches='tight')

    print("Training Results:")
    print("Accuracy:", acc_list)
    print("Loss:", lost_list)

    ''' Implement transmission by smart contract '''
    print("\nImplementing transmission by smart contract...")

    # Calculate and transfer the reward to each client, then present the results
    reward = []
    new_balance = []
    for i in range(conf['client_total_num']):
        reward.append(ctt.functions.reward_calculate(
            i, conf["client_total_num"]).call({"from": w3.eth.accounts[i]}))
        ctt.functions.reward_transfer(reward[i]).transact(
            {"from": w3.eth.accounts[i]})

        if deposits[i] > reward[i]:
            print("Account", i, "is charged", deposits[i] - reward[i],
                  "gas, with balance:", w3.eth.get_balance(w3.eth.accounts[i]))
        else:
            print("Account", i, "get the deposit",
                  reward[i] - deposits[i], "gas, with balance:", w3.eth.get_balance(w3.eth.accounts[i]))
        new_balance.append(round(w3.eth.get_balance(w3.eth.accounts[i]), 2))
