# RetinAI: AI Assisted Diabetic Retinopathy Image Screening

This project utilizes Vision Language Models (VLMs) to create an accessible and accurate screening tool for diabetic retinopathy. By analyzing retinal images, the system will assist healthcare providers in underserved areas with early detection of the disease, helping to prevent irreversible vision loss.

***

### Repository Structure
```bash
retinai/
├── data/              # Raw and processed retinal image datasets
├── notebooks/         # Jupyter notebooks for data exploration and prototyping
│   └── setup.ipynb
├── src/               # Source code for data loading, model training, and inference
├── ui/                # Code for the user interface
├── results/           # Early outputs and visualizations
├── docs/              # Project documentation, diagrams, and reports
├── .gitignore         # Specifies files to be ignored by Git
├── README.md          # Project overview and setup instructions
└── requirements.txt   # List of all project dependencies
```

***

### Installation and Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/saririans/RetinAI.git
    cd retinai
    ```

2.  **Create and Activate a Virtual Environment**:

    ```bash
    # Conda
    conda create -n <VENV> python=3.10
    conda activate <VENV>

    # Or venv
    python -m venv <VENV>
    source <VENV>/bin/activate  
    ```

3.  **Install Required Libraries**:
    Once environment is active, install all necessary libraries from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

***




