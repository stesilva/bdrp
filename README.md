This repository provides implementations, data utilities, and experiment scripts for research involving Relational Graph Convolutional Networks (RGCNs) and other graph-based learning models to implement edge weights into RGCN.

📌 **Key Features**

🧠 **Regular RGCN Baseline**  
Located in `baselines/standard_rgcn/`, this module includes the standard Relational Graph Convolutional Network implementation: the primary baseline for all experiments.

⚗️ **Five Experiment Methods**  
Found inside `experiments/`, these are the five distinct methods described in the project report.  
Each subfolder contains its own scripts, configurations, and detailed instructions.

🛠 **Modular Core System**  
Shared logic is organized into:

- `core/`: Model definitions and training logic  
- `utils/`: Data processing, logging, and evaluation tools

📁 **Repository Structure**

bdrp/

  ├── baselines/ # Regular RGCN baseline implementation
  
  │ └── standard_rgcn/
  
  ├── core/ # Core framework modules
  
  ├── data/ # Dataset files or processing helpers
  
  ├── docs/
  
  │ └── visualization/
  
  ├── experiments/ # Five methods used in the report
  
  ├── scripts/ # Miscellaneous scripts for automation
  
  ├── utils/ # Utility functions and helpers
  
  ├── requirements.txt # Dependencies list
  
  └── .gitignore


🚀 **Getting Started**

**1️⃣ Clone the repository**
git clone https://github.com/stesilva/bdrp.git
cd bdrp


**2️⃣ Install dependencies**

Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate

Then install required packages:
pip install -r requirements.txt


🧪 **Running Experiments**

📍 **Baseline: Regular RGCN**  
Path:  
`baselines/standard_rgcn/`  

Run it using scripts provided in this directory.

📍 **Five Custom Methods (from the Report)**  
Path:  
`experiments/`  

Each subfolder represents one experiment setup.  
To execute a method, navigate to its directory and follow the included README.

Some methods may require:
- Preparing custom datasets  
- Selecting configuration files  
- Saving outputs into `logs/` or `results/`





