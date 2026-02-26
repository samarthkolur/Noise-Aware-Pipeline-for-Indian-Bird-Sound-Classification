Before running Python:
-> run this "source venv/bin/activate"
If you forget this, you'll get error 



📦 Dataset Setup (IBC53)

This project uses the IBC53 – Indian Bird Call Dataset.

Step 1: Install Kaggle CLI

After activating your virtual environment:

pip install kaggle
Step 2: Configure Kaggle API

Go to https://www.kaggle.com/account

Click “Create New API Token”

Download kaggle.json

Move it to:

mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
Step 3: Download Dataset

From project root:

kaggle datasets download -d arghyasahoo/ibc53-indian-bird-call-dataset

Unzip:

unzip ibc53-indian-bird-call-dataset.zip -d data/

Final structure should look like:

BirdNetProject/
│
├── data/
│   └── IBC53/
│       ├── Corvus_splendens/
│       ├── ...

# Setup Instructions

1. Clone repository
2. Create virtual environment

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Download dataset (see Dataset Setup section)