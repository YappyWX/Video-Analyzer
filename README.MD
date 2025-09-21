# Installation Guide

First, Clone the repository

`git clone https://github.com/YappyWX/Video-Analyzer`

*When git clones successfully, open 2 terminals (1 for client, other for server)*

### Terminal 1 (Client-Side)

```
cd client
npm install
```

### Terminal 2 (Server-Side)

```
cd server
python -m venv [venvName]
source venv/Scripts/activate    (Windows)
source venv/bin/activate        (MacOS/Linux)
pip install -r requirements.txt
```

# Running the Application
After succesfully installing the packages for server and client

### Terminal 1 (Client-Side)
```
npm run dev
```

After running, view terminal to get application link

### Terminal 2 (Server-Side)
```
python main.py
```
