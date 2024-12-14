// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Federated Learning Contract
/// @notice This contract handles client registration and token management for federated learning.
/// @dev Deployment script is located in `scripts/deploy.js`.
contract FederatedLearning {
    mapping(address => uint256) public balances;
    uint256 public constant TOKEN_REWARD = 100;

    /// @notice Registers a client and rewards them with tokens.
    function registerClient() public {
        balances[msg.sender] += TOKEN_REWARD;
    }

    /// @notice Retrieves the balance of a client.
    /// @param client The address of the client.
    /// @return The token balance of the client.
    function getBalance(address client) public view returns (uint256) {
        return balances[client];
    }

    /// @notice Deducts a specified amount of tokens from a client's balance.
    /// @param client The address of the client.
    /// @param amount The amount of tokens to deduct.
    function useGas(address client, uint256 amount) public {
        require(balances[client] >= amount, "Insufficient balance");
        balances[client] -= amount;
    }
}
