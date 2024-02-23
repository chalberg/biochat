# BioChat

A chatbot for QA using biomedical language and documents.

## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Liscense](#liscense)

## About <a name="about"></a>

## Getting Started <a name="getting-started"> </a>

#### Cloning the Repository
Download this repository using the following commands:
```bash
$ git clone https://github.com/chalberg/biochat.git
$ cd biochat
```

#### Setting up a Virtual Environment with Conda
Because of the size of the dependencies, it is highly reccomended that a virtual environment is set up for this project. Navigate to this project directory and use these commands to setup a venv with Conda:
```bash
conda create --name biochat
conda activate biochat
conda install --file requirements.txt
```
If there are issues with installing requirements, ensure that conda-forge is configured:
```bash
conda config --add channels conda-forge
```
If issues persist, try installing with Pip:
```bash 
conda install pip
pip install -r requirements.txt
```
On Windows systems, Chromadb may have additional requirements for installation. Follow [this guide](https://github.com/bycloudai/InstallVSBuildToolsWindows) if you get the following error:
```
 error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

## Usage <a name="usage"></a>

#### Launching the App
The main feature of this repository is a web app which allows you to  interact with the chatbot. To launch the app, use the following command while in the project directory:
```bash
python streamlit run app.py
```

## Liscense <a name="liscense"></a>

MIT License

Copyright (c) [2024] [Charlie Halberg]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.