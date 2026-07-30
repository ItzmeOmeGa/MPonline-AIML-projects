Here is all the content provided as plain raw Markdown text inside code blocks, so you can copy and paste each document without any nested formatting issues.

---

### Master `README.md` (Root Directory)

```markdown
# Multi-Domain Deep Learning & Reinforcement Learning Repository (PyTorch)

A production-ready collection of Deep Learning and Reinforcement Learning projects built with **PyTorch**, **Gymnasium**, and **Scikit-Learn**. Each project is fully modularized with clean code separation for dataset management, model definitions, training engines, evaluation metrics, and inference pipelines.

---

## 📌 Projects Included

1. **Face Recognition in the Wild (LFW Dataset)**
2. **Brain Tumor & Cancer Detection via MRI Scans**
3. **Hybrid Movie Recommendation System**
4. **Cart-Pole Control Agent (DQN)**
5. **Lunar Lander Precision Landing Agent (DQN)**

---

## 📁 Repository Structure

```text
.
├── 1_lfw_face_recognition/
│   ├── dataset.py        # LFW dataset fetcher, filtering & transforms
│   ├── model.py          # Custom Deep Face CNN with BatchNorm & Dropout
│   ├── utils.py          # Loss/Acc plots & per-person Confusion Matrix
│   ├── train.py          # Main training loop with checkpointing
│   ├── predict.py        # Single-image face identification CLI
│   └── README.md         # Project documentation
│
├── 2_mri_cancer_detection/
│   ├── dataset.py        # Custom MRI dataset loader & medical transforms
│   ├── model.py          # CNN with Spatial Attention module
│   ├── utils.py          # Clinical metrics (ROC-AUC, Precision, Recall)
│   ├── train.py          # Training loop with Learning Rate Decay
│   ├── predict.py        # Single MRI scan diagnosis CLI
│   └── README.md         # Project documentation
│
├── 3_movie_recommendation_system/
│   ├── dataset.py        # MovieLens dataset fetcher & tensor mappings
│   ├── model.py          # Deep Matrix Factorization Neural Network
│   ├── utils.py          # Evaluation metrics (RMSE/MAE) & Cold-Start filter
│   ├── train.py          # Embedding training engine
│   ├── predict.py        # Top-N user recommendations CLI
│   └── README.md         # Project documentation
│
├── 4_cartpole_rl/
│   ├── model.py          # Q-Network architecture
│   ├── memory.py         # Experience Replay Buffer
│   ├── agent.py          # DQNAgent with target network & Epsilon-Greedy
│   ├── utils.py          # Reward progress plotter with moving averages
│   ├── train.py          # Gymnasium CartPole-v1 training loop
│   ├── evaluate.py       # Rendered environment agent test execution
│   └── README.md         # Project documentation
│
└── 5_lunar_lander_rl/
    ├── model.py          # Deep Q-Network for continuous state vector
    ├── memory.py         # Large-capacity Experience Replay Buffer
    ├── agent.py          # DQN Agent with Soft Target Updates
    ├── utils.py          # Convergence tracker & moving average plots
    ├── train.py          # Gymnasium LunarLander-v3 training loop
    ├── evaluate.py       # Rendered environment precision landing test
    └── README.md         # Project documentation

```

---

## 🛠️ Requirements & Installation

Clone the repository and install all required dependencies:

```bash
git clone [https://github.com/your-username/pytorch-dl-rl-projects.git](https://github.com/your-username/pytorch-dl-rl-projects.git)
cd pytorch-dl-rl-projects

# Install all prerequisites
pip install torch torchvision "gymnasium[box2d]" scikit-learn pandas matplotlib seaborn pillow numpy

```

---

## 🚀 Projects Overview & Usage

### 1. LFW Face Recognition

Recognizes individual human faces in unconstrained environments using custom CNNs and `sklearn.datasets.fetch_lfw_people`.

* **Train:** `python 1_lfw_face_recognition/train.py`
* **Inference:** `python 1_lfw_face_recognition/predict.py path/to/face.jpg`

### 2. MRI Brain Cancer Detection

Classifies brain tumor scans using a CNN equipped with a **Spatial Attention Module** to highlight localized anomalous tissues.

* **Train:** `python 2_mri_cancer_detection/train.py`
* **Inference:** `python 2_mri_cancer_detection/predict.py path/to/mri.jpg`

### 3. Movie Recommendation System

A hybrid engine combining **Collaborative Filtering** (Deep Matrix Factorization via latent user/item embeddings) with **Content-Based Filtering** (genre cosine similarity for cold-start users).

* **Train:** `python 3_movie_recommendation_system/train.py`
* **Inference:** `python 3_movie_recommendation_system/predict.py <USER_ID>`

### 4. Cart-Pole Control (DQN)

Trains an RL agent using Deep Q-Learning, Experience Replay, and Target Networks to balance a pole on a cart in `CartPole-v1`.

* **Train:** `python 4_cartpole_rl/train.py`
* **Visualize:** `python 4_cartpole_rl/evaluate.py`

### 5. Lunar Lander Landing Agent (DQN)

Controls a 2D lunar lander to perform soft precision landings between landing flags using PyTorch and `Gymnasium` Box2D environment physics.

* **Train:** `python 5_lunar_lander_rl/train.py`
* **Visualize:** `python 5_lunar_lander_rl/evaluate.py`

---

## 📊 Shared Design Principles

* **Modularity:** Separates data loading (`dataset.py`), model structure (`model.py`), evaluation tools (`utils.py`), training loops (`train.py`), and inference pipelines (`predict.py`/`evaluate.py`).
* **Checkpointing:** Automatically saves the best model state dict based on validation loss/rewards.
* **Evaluation:** Displays loss/accuracy curves, confusion matrices, RMSE scores, or reward moving averages.

---

## 📜 License

This repository is licensed under the MIT License. Feel free to adapt and expand for your machine learning projects!

```

---

### Project 1: `lfw_face_recognition/README.md`

```markdown
# Face Recognition in the Wild (LFW Dataset)

A PyTorch-based Deep Convolutional Neural Network (CNN) engineered for face recognition on the **Labeled Faces in the Wild (LFW)** dataset. This implementation handles unconstrained real-world facial images featuring variations in pose, lighting, expression, and background.

---

## 📁 Project Structure

```text
lfw_face_recognition/
│
├── dataset.py        # Dataset fetching, filtering, and facial image augmentations
├── model.py          # Custom Deep CNN with BatchNorm & Dropout regularizations
├── utils.py          # Training curve plotting & per-person Confusion Matrix
├── train.py          # Training loop with adaptive learning rates and checkpointing
├── predict.py        # Single image face recognition CLI script
└── README.md         # Project documentation

```

---

## ✨ Key Features

* **Real-World Augmentation:** Uses horizontal flips, random rotations, and color jitter to build robustness against varying real-world lighting conditions.
* **Class Balancing:** Filters classes using a configurable threshold (`min_faces_per_person=70`) to handle dataset imbalance.
* **Modern Architecture:** Employs dual-conv blocks with `BatchNorm2d`, `Dropout`, and `AdaptiveAvgPool2d` to extract facial feature representations effectively.
* **Detailed Evaluation:** Generates per-person Precision, Recall, F1-Scores, and a Seaborn Confusion Matrix heatmap.

---

## 🛠️ Usage

### 1. Train the Model

```bash
python train.py

```

*Automatically downloads LFW dataset via Scikit-Learn, trains the network for 25 epochs, and saves the best model state to `best_lfw_model.pth`.*

### 2. Predict on a Face Image

```bash
python predict.py path/to/sample_face.jpg

```

```text
Identified Person: George W Bush (94.21% confidence)

```

```

---

### Project 2: `mri_cancer_detection/README.md`

```markdown
# Brain Tumor & Cancer Detection via MRI Scans

A PyTorch deep learning framework for brain cancer diagnosis and tissue classification using Magnetic Resonance Imaging (MRI) scans. The model incorporates a **Spatial Attention Module** to help the network focus specifically on anomalous tissue regions.

---

## 📁 Project Structure

```text
mri_cancer_detection/
│
├── dataset.py        # Custom dataset loader for folder-based MRI directories
├── model.py          # CNN Architecture with integrated Spatial Attention Module
├── utils.py          # Clinical evaluation metrics (ROC-AUC, Precision, Recall)
├── train.py          # Training engine with ReduceLROnPlateau & Checkpointing
├── predict.py        # Diagnosis inference pipeline for individual MRI scans
└── README.md         # Project documentation

```

---

## ✨ Key Features

* **Spatial Attention:** Features a custom 2D Spatial Attention block that dynamically highlights abnormal lesion boundaries.
* **Medical-Grade Data Pipeline:** Uses affine and brightness transforms tailored for MRI scans while preserving clinical anatomical orientation.
* **Clinical Metrics:** Evaluates diagnostic capability using Receiver Operating Characteristic Area Under the Curve (**ROC-AUC**), Sensitivity (Recall), and Specificity.

---

## 🛠️ Dataset Setup & Usage

### 1. Dataset Directory Setup

Organize your MRI scans into class subfolders:

```text
data/mri_scans/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

```

### 2. Train the Model

```bash
python train.py

```

### 3. Diagnose an MRI Scan

```bash
python predict.py path/to/mri_scan.jpg

```

```text
Diagnosis: GLIOMA (98.65% confidence)

```

```

---

### Project 3: `movie_recommendation_system/README.md`

```markdown
# Hybrid Movie Recommendation System

A PyTorch Deep Matrix Factorization engine combined with Content-Based Filtering for predicting user movie ratings and serving personalized top-N recommendations.

---

## 📁 Project Structure

```text
movie_recommendation_system/
│
├── dataset.py        # MovieLens dataset fetcher & tensor mappings
├── model.py          # Embedding-based Deep Matrix Factorization network
├── utils.py          # Metrics (RMSE/MAE) & Cold-Start Content-Based Engine
├── train.py          # Latent factor training loop
├── predict.py        # Top-N user recommendation CLI
└── README.md         # Project documentation

```

---

## ✨ Key Features

* **Collaborative Filtering:** Learns joint latent embedding spaces for users and movies alongside user and item biases.
* **Cold-Start Fallback:** Integrates genre Cosine Similarity content-filtering to recommend top titles for unrated items or new users.
* **Automatic Dataset Management:** Downloads and processes the MovieLens-100k dataset seamlessly on the fly.
* **Quantitative Evaluation:** Evaluates recommendation accuracy using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

---

## 🛠️ Usage

### 1. Train the Recommender Engine

```bash
python train.py

```

*Trains user/movie embeddings for 12 epochs and outputs validation RMSE/MAE metrics.*

### 2. Get Top Recommendations

```bash
# Get top 5 recommendations for User ID 196
python predict.py 196

```

```text
--- Top 5 Movie Recommendations for User 196 ---
 Movie ID                      Title  Predicted Rating
      174            Raiders of the Lost Ark (1981)              4.68
      318  Shawshank Redemption, The (1994)              4.62
      ...

```

```

---

### Project 4: `cartpole_rl/README.md`

```markdown
# Cart-Pole Control Agent using Deep Q-Networks (DQN)

A Reinforcement Learning (RL) agent trained with **Deep Q-Learning (DQN)** in PyTorch to balance a pole upright on a moving cart using **Gymnasium**'s `CartPole-v1` environment.

---

## 📁 Project Structure

```text
cartpole_rl/
│
├── model.py          # Deep Q-Network MLP architecture
├── memory.py         # Experience Replay Buffer for sample efficiency
├── agent.py          # DQNAgent with target network and Epsilon-Greedy policy
├── utils.py          # Training curve plotter with rolling average curves
├── train.py          # Gymnasium CartPole-v1 training execution loop
├── evaluate.py       # Visual environment rendering script
└── README.md         # Project documentation

```

---

## ✨ Key Features

* **Experience Replay Buffer:** Stores $(s, a, r, s', \text{done})$ transitions to break temporal dependencies during gradient updates.
* **Target Q-Network:** Stabilizes learning by maintaining a separate, periodically updated network for calculating temporal difference targets.
* **Epsilon-Greedy Exploration:** Smoothly decays exploration rate ($\epsilon$) from fully random actions to greedy policy execution.

---

## 🛠️ Usage

### 1. Train the Agent

```bash
python train.py

```

*Trains until the environment is solved (20-episode moving average reward $\ge 490$) and saves `cartpole_dqn.pth`.*

### 2. Render & Test Agent

```bash
python evaluate.py

```

*Opens a rendered GUI window displaying the trained agent balancing the pole in real time.*

```

---

### Project 5: `lunar_lander_rl/README.md`

```markdown
# Lunar Lander Precision Landing Agent (DQN)

A Deep Q-Network (DQN) Reinforcement Learning agent trained using **PyTorch** and **Gymnasium** (`LunarLander-v3`/`v2`) to perform soft landings on a moon pad using main and side thruster controls.

---

## 📁 Project Structure

```text
lunar_lander_rl/
│
├── model.py          # Q-Network for 8D continuous state inputs
├── memory.py         # High-capacity Experience Replay Buffer (100,000 steps)
├── agent.py          # DQN Agent with Soft Target Updates (Polyak Averaging)
├── utils.py          # Training history visualization utilities
├── train.py          # Training loop with early stopping upon benchmark completion
├── evaluate.py       # Rendered environment landing execution script
└── README.md         # Project documentation

```

---

## ✨ Key Features

* **8D State Vector Processing:** Maps continuous flight dynamics (position, velocity, angle, angular speed, and ground contact indicators) to 4 discrete thruster commands.
* **Soft Target Network Updates:** Uses Polyak averaging ($\tau = 0.001$) for target network weight updates.
* **Benchmark Solved Criteria:** Automatically detects when the agent hits the official benchmark score ($\ge +200$ points averaged over 100 consecutive episodes).

---

## 🛠️ Usage

### 1. Install Requirements

```bash
pip install "gymnasium[box2d]" torch matplotlib numpy

```

### 2. Train the Agent

```bash
python train.py

```

### 3. Visually Render Lunar Landings

```bash
python evaluate.py

```
