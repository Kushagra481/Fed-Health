Fed-Health: A Federated Learning System for Privacy-Preserving Medical Image Classification
Introduction
In the critical field of healthcare, machine learning models hold immense potential for improving diagnostics and patient outcomes. However, training these powerful models often requires vast amounts of data, which, in the medical context, is highly sensitive and subject to strict privacy regulations like HIPAA. Centralizing patient data from multiple hospitals for model training is often not feasible due to these privacy concerns and legal restrictions.

Fed-Health offers a robust solution to this challenge by implementing a Federated Learning system for medical image classification. This project demonstrates how a deep learning model can be collaboratively trained across multiple simulated "hospitals" (clients), each holding its own private Chest X-Ray images, without ever sharing the raw patient data. Only aggregated model updates are exchanged, ensuring patient privacy while still benefiting from a diverse, larger dataset.

Features
Privacy-Preserving Training: Trains a CNN model for pneumonia detection without centralizing sensitive patient X-ray images.

Simulated Multi-Client Environment: Mimics a real-world scenario with multiple independent "hospitals" acting as clients.

Non-IID Data Simulation: Intentionally distributes data unevenly among clients to reflect realistic data heterogeneity, a key challenge in federated learning.

Performance Benchmarking: Compares the federated model's performance against a traditionally trained centralized model to demonstrate effectiveness.

Comprehensive Visualizations: Provides insights into training progress, model accuracy, and loss over communication rounds.

Modular and Extensible: Designed with clear separation of concerns for easy understanding and future enhancements.

How It Works (Conceptual Overview)
The Fed-Health system operates on the principles of Federated Averaging (FedAvg):

Global Model Initialization: A central server initializes a convolutional neural network (CNN) model.

Model Distribution: The server sends the current global model to a selected subset of participating clients (simulated hospitals).

Local Training: Each client trains the received model using its own private, local Chest X-Ray dataset. The raw data never leaves the client's device.

Update Transmission: After local training, each client sends only its updated model parameters (weights and biases) back to the central server.

Global Aggregation: The central server aggregates these received model updates (e.g., by averaging them) to create an improved global model.

Iteration: This process repeats for several communication rounds, iteratively refining the global model to achieve high accuracy while preserving data privacy.

Dataset
This project utilizes the Chest X-Ray Images (Pneumonia) dataset, publicly available on Kaggle. This dataset contains X-ray images categorized as either 'Normal' or 'Pneumonia'.

We preprocess these images (resizing, normalization) and then strategically partition them among our simulated clients to create a non-IID (non-identically and independently distributed) data environment. This means that each client's local dataset will have a different distribution of 'Normal' and 'Pneumonia' cases, accurately reflecting the varied patient populations and diagnostic focuses of different hospitals.

Results & Visualizations
The notebooks/ directory contains Jupyter notebooks to help you explore the data, run the simulation, and visualize the results. You'll be able to see:

Training Progress: Plots showing the global model's accuracy and loss over each federated communication round.

Centralized vs. Federated Performance: A direct comparison of the final model's accuracy and other metrics when trained centrally versus using our federated approach.

Confusion Matrix: A visualization of the federated model's classification performance (True Positives, False Positives, etc.).

These visualizations will highlight how the federated learning approach can achieve competitive performance to centralized training while maintaining data privacy.

Getting Started
Follow these steps to set up and run the Fed-Health simulation on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

Installation
Clone the repository:

git clone https://github.com/your-username/Fed-Health-FL.git
cd Fed-Health-FL

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Download the Dataset:

Download the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle.

Extract the contents into the data/chest_xray/ directory within your project. The structure should look like data/chest_xray/train, data/chest_xray/test, data/chest_xray/val.

Running the Simulation
The simulation involves starting the central server and then launching multiple client processes.

Start the Federated Learning Server:
Open a new terminal and run:

python server/main.py

The server will start and wait for clients to connect.

Start the Clients:
Open separate terminals for each client you want to simulate (e.g., 3-5 terminals for a small simulation). In each new terminal, run:

python clients/main.py

Each client will connect to the server, receive the global model, perform local training, and send updates back.

Alternatively, you can use the provided shell script to run multiple clients:

# Make the script executable
chmod +x scripts/run_simulation.sh
# Run the script (adjust number of clients if needed inside the script)
./scripts/run_simulation.sh

The server console will display the progress of the federated training rounds, including metrics like accuracy and loss.

Project Structure
Fed-Health-FL/
├── README.md
├── requirements.txt
├── server/
│   ├── main.py        # Central server orchestration
│   └── model.py       # Defines the CNN model architecture
├── clients/
│   ├── main.py        # Client-side logic for local training
│   ├── model.py       # Same CNN model definition as server
│   └── utils.py       # Data loading, preprocessing, and non-IID partitioning
├── data/
│   └── chest_xray/    # Extracted Chest X-Ray dataset goes here
├── notebooks/
│   ├── EDA_and_Data_Prep.ipynb  # Explore the dataset and preprocessing steps
│   └── Training_Visualizations.ipynb # Visualize FL training metrics and comparisons
└── scripts/
    └── run_simulation.sh # Helper script to run multiple clients

Future Work
This project can be extended in several exciting ways:

Implement Advanced FL Algorithms: Explore algorithms beyond FedAvg, such as FedProx, SCAFFOLD, or personalized federated learning techniques.

Integrate Privacy-Enhancing Technologies (PETs): Add layers like Secure Aggregation or Differential Privacy to further strengthen privacy guarantees.

Real-time Client Simulation: Develop a more sophisticated client simulation that can join and leave the federation dynamically.

UI for Monitoring: Create a simple web-based dashboard to visualize the federated training process in real-time.

Deployment to Edge Devices: Explore deploying clients to actual edge devices (e.g., Raspberry Pi) for a more realistic setup.

Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
