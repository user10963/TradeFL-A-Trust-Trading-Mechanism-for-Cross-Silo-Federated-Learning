from solcx import set_solc_version
from solcx import compile_source

set_solc_version('v0.8.0')


def compile_source_file(file_path):
    ''' Compile the contract '''
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    return compile_source(source)


def deploy_contract(w3, contract_interface):
    ''' Deploy the contract '''
    account = w3.eth.accounts[0]

    contract = w3.eth.contract(
        abi=contract_interface['abi'],
        bytecode=contract_interface['bin'])
    tx_hash = contract.constructor().transact({'from': account})

    address = w3.eth.get_transaction_receipt(tx_hash)['contractAddress']
    return address
