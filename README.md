# MAIT Group 2 - Sensing the Drift Challenge

This repository contains the work of **MAIT Group 2** for the private Kaggle competition **"Sensing the Drift."**

## Project Overview

This project focuses on developing virtual sensor models to predict vehicle sideslip angle (VSA) while addressing data imbalance challenges. The goal is to create accurate VSA estimates for Advanced Driver-Assistance Systems (ADAS) using standard sensor data instead of expensive dedicated sensors.

For more information about the topic, please visit this [link](https://gemini.google.com/share/fd9631a2ced2)

## Team Members

**MAIT Group 2**
- Member 1: [Name]
- Member 2: [Name]
- Member 3: [Name]
- Member 4: [Name] *(if applicable)*

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. Clone this repository:
   ```bash
   git clone <your-gitlab-repo-url>
   cd MAIT_G2_Drift_Challenge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* Install the Kaggle CLI and download the dataset:
   ```bash
   pip install kaggle
   kaggle competitions download -c sensing-the-drift
   ```
   
   See the [Kaggle API docs](https://www.kaggle.com/docs/api) for setup details.

## Repository Structure

```
MAIT_G2_Drift_Challenge/
├── datasets_challenge/
│   ├── train/              # Training data (train_part_001.csv, ...)
│   ├── test/               # Test data (test_part_001.csv, ...)
│   └── sideSlip_distribution.csv
├── utils/                  # Utility scripts and helper functions
├── notebooks/              # Jupyter notebooks for experiments
├── models/                 # Trained models and checkpoints
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

**Note:** The `datasets_challenge` folder should be populated with the competition data after downloading from Kaggle.

## Kaggle Competition

**Join the challenge:**
[https://www.kaggle.com/t/c6b20902477e482c94400773be035fa2](https://www.kaggle.com/t/c6b20902477e482c94400773be035fa2)

**⚠️ Confidentiality Notice:**
This is a **private** challenge. The invitation link should **not be shared** outside the team. The dataset is for competition use only and **must not be redistributed or posted publicly**. Sharing the link or data outside the competition may result in removal from the case study/guided project.

## Project Goals

- Develop virtual sensor models for vehicle sideslip angle prediction
- Address data imbalance in real-world driving datasets
- Explore techniques: cost-sensitive losses, resampling, synthetic data, drift-focused augmentation
- Submit weekly predictions to the Kaggle leaderboard
- Deliver a real-time sideslip estimator (TorchScript or equivalent)

## Deliverables

- Weekly leaderboard submissions
- Well-documented code in this repository
- Demo of real-time sideslip estimator
- Final presentation (15-20 minutes, ~5 min per person)
- Final report (~4 pages, paper-like format) *for Digital Science students*

## Grading (MAIT)

- 70% Documentation + Demo
- 30% Presentation
- Bonus Points based on Leaderboard performance

## Timeline

- **Regular Meetings:** Tuesdays 2pm-4pm in Room 0503
- **Final Presentation:** 10.02.2026
- **Final Submission:** 28.02.2026

## Supervisors

- [Noah Pütz](https://www.th-koeln.de/personen/noah_christoph.puetz/)
- [Prof. Dr. Thomas Bartz-Beielstein](https://www.th-koeln.de/personen/thomas.bartz-beielstein/)

## Progress Tracking

- [ ] Initial setup and environment configuration
- [ ] Data exploration and analysis
- [ ] Baseline model implementation
- [ ] Address data imbalance
- [ ] Model optimization and experiments
- [ ] Real-time estimator demo
- [ ] Final presentation preparation
- [ ] Documentation and report

## License

This project is for educational purposes only as part of the MAIT program at TH Köln.
