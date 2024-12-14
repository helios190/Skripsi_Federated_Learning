const { ethers } = require("hardhat");

async function main() {
    const [deployer] = await ethers.getSigners();
    console.log("Deploying contracts with the account:", deployer.address);

    const FederatedLearning = await ethers.getContractFactory("FederatedLearning");
    const federatedLearning = await FederatedLearning.deploy();
    await federatedLearning.deployed();

    console.log("FederatedLearning contract deployed to:", federatedLearning.address);
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
