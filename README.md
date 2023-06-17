# Steps to run API locally
1. Download the repo
    1. Preferable to use `git pull` as the file `features/feats.pkl` was uploaded using Git LFS. Standard downloading as ZIP might not download it. Although that file isn't necessary for the API.
2. Get into the repo
3. Initialise a Virtual Environment (optional)
    1. Use `python -m venv <name>` to create virtual environment
    2. Get into virtual environment
        1. On MacOS/Linux: `source <name>/bin/activate`
        2. On Windows: `<name>\Scripts\activate`
4. Install packages/dependencies `pip install -r requirements.txt`
5. Run server using `python app.py`
6. Copy the IP Address start with `192.` to the App's `...` button beside the `Upload Audio` button (**IMPORTANT**)
