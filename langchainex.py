import hashlib
import json

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'proof': proof,
            'previous_hash': previous_hash
        }
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def hash(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

# Example usage:

# Instantiate the blockchain
blockchain = Blockchain()

# Create a new block
blockchain.create_block(proof=2, previous_hash=blockchain.hash(blockchain.get_previous_block()))

# Print the blockchain
print(blockchain.chain)

#-----------------------------------------------------------------#
#-----------------------------------------------------------------#


import hashlib
import json
from time import time

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        """
        Create a new block in the blockchain

        :param proof: <int> The proof of work for the new block
        :param previous_hash: (Optional) <str> Hash of previous block
        :return: <dict> New block
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []
        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount):
        """
        Creates a new transaction to go into the next mined Block

        :param sender: <str> Address of the Sender
        :param recipient: <str> Address of the Recipient
        :param amount: <int> Amount
        :return: <int> The index of the block that will hold this transaction
        """
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block['index'] + 1

    @staticmethod
    def hash(block):
        """
        Creates a SHA-256 hash of a block

        :param block: <dict> Block
        :return: <str> Hash
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

# Example usage:

# Instantiate the blockchain
blockchain = Blockchain()

# Add a transaction
blockchain.new_transaction("Alice", "Bob", 5)

# Mine a new block
last_block = blockchain.last_block
last_proof = last_block['proof']
proof = 12345  # For example purposes, replace this with a real proof of work algorithm
previous_hash = blockchain.hash(last_block)
block = blockchain.create_block(proof, previous_hash)

# Print the blockchain
print(blockchain.chain)
