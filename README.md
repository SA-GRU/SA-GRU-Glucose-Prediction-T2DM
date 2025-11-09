# SA-GRU-Glucose-Prediction-T2DM
The repository of the SA-GRU model included in the paper we submitted to the journal. Currently, it contains code for data preprocessing, data splitting, and pseudo-code. The complete training code and data splitting will be updated after the paper is accepted.

## 🔍 Repository Overview
To balance reproducibility and intellectual property protection (avoiding content leakage before acceptance), we provide **phased public materials** as follows:
- Currently available: Preprocessing code, data splitting code, pseudocode, and flow diagrams (fully align with the manuscript's Methods section).
- To be updated upon acceptance: Complete model training code, full dataset splitting files, and detailed reproduction instructions.

## 📂 File Structure

├── config.py                       # Core parameter configuration (fully released post-acceptance)
├── module1_preprocessing.py        # Module 1: Data preprocessing (Section 2.2)
├── module2_data_split.py           # Module 2: Data splitting (Section 2.5)
├── module3_pseudocode_flowchart.py # Module 3: Pseudocode + Flowchart
├── normalization_params.json       # Normalization parameter example
├── LICENSE                         # MIT License
└── README.md                       # Documentation

🚀 Quick Start
1. Dependencies
bash
pip install pandas numpy scikit-learn
2. Key Notes
Core parameters (conversion factor, window size, split ratio, etc.) are stored in config.py, which is not fully released yet.
Code execution relies on config.py — the scripts will not run without this file.
Input data requirement: CSV file containing patient_id, glucose_value (mmol/L), and time (hh:mm).
3. Run Preprocessing (Module 1)
bash
python module1_preprocessing.py
Reads raw data and outputs preprocessed time-step data + normalization parameters.
Core steps: Unit conversion → time formatting → normalization → time-step construction.
4. Run Data Splitting (Module 2)
bash
python module2_data_split.py
Reads preprocessed data and outputs 10 repeated split results.
Key principle: Patient-stratified splitting to avoid data leakage.
5. View Pseudocode & Flowchart (Module 3)
bash
python module3_pseudocode_flowchart.py
Contains full workflow logic and visualization, covering preprocessing, splitting, model training, and evaluation.

⏰ Post-Acceptance Update Plan
After the manuscript is accepted, we will update this repository, which includes:
The complete code of the paper. Detailed re-enactment guide (environment configuration, code execution steps, expected results).
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
📧 Contact
For questions about the code or manuscript, please contact: xuejian464@163.com
