# **Mixture-of-Experts (MoE) Subnet**

This repository contains the full implementation of the **MoE Subnet**, including the miner, validator, routing logic, expert modules, and all supporting infrastructure.
The subnet is designed to support distributed training and inference across a decentralized network, with miners running experts and validators ensuring correctness, consistency, and performance.

---

## 🌐 **What’s Inside the Subnet**

📄 **Detailed subnet overview:**
👉 *[What is in the subnet](https://docs.google.com/document/d/1a-ZFex1KbJrHkfq4SoxWLqQYqNlPAyTI69-PwnCWwyI/edit?usp=sharing)*

---

## 🚀 **Getting Started**

### **Requirements**

* Python 3.10+
* Additional packages in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⛏️ **Guide: Running a Miner**

Miner nodes host experts and execute workloads assigned by the router/validator layer.

Follow the full guide here:
👉 *[Guide to run a miner](https://github.com/CrucibleAILabs/subnet-MoE/blob/main/mycelia/miner/README.md)*

The guide covers:

* [missing] Registering your miner
* Running the miner process
* [missing] Monitoring expert load
* [add more]
---

## 🛡️ **Guide: Running a Validator**

Validators verify miner outputs, compute scores, and stabilize the subnet.

Full instructions here:
👉 *[Guide to run a validator](https://github.com/CrucibleAILabs/subnet-MoE/blob/main/mycelia/validator/README.md)*

The guide includes:

* [missing] Validator architecture
* How to run a validator node
* [missing] Scoring logic
* [add more]

---

## 📦 **Project Structure**

```
moe-subnet/
├── miner/ # modules specifig to miner
├── validator/ # modules specifig to validator
├── shared/ # modules used by both miner and validator
└── README.md
```

---

## 👥 **Contributing**

Contributions are welcome!
If you're adding a new expert, router, or validator behavior, please open a PR.

---

## 📄 **License**


---

