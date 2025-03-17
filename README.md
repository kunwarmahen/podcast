# Empire Podcast Portal

Welcome to the **Empire Podcast Portal**! This application allows you to listen, converse, and get summary and important tidbits about the a particular episode of a podcas.

## Features

- Host and manage your podcasts
- Automatic transcription using Whisper
- User-friendly web interface

## Prerequisites

Ensure you have the following installed on your system before proceeding:

- Python 3.x
- Virtual environment (`venv`)
- Pip (Python package manager)

## Deployment Guide

Follow these steps to deploy the Empire Podcast Portal on your local system or server.

### 1. Clone the Repository

Clone the repository from GitHub to access the code:

```bash
git clone https://github.com/kunwarmahen/podcast
cd podcast
```

### 2. Set Up a Virtual Environment

Create a virtual environment to manage project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use venv\Scripts\activate
```

### 3. Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_audio.txt
```

### 4. Download Whisper Model

Whisper will be used for transcribing podcasts. Ensure you have the model installed:

> **Note:** You may need to log in to [Hugging Face](https://huggingface.co/) to download the `seasme` model.

### 5. Run the Application

Start the application with the following command:

```bash
python app.py
```

### 6. Access the Portal

Open your web browser and go to:

```
http://localhost:5000
```

Enjoy your podcast portal!

#### 1. Download any podcast or listen to it.

![Screenshot from 2025-03-16 19-16-50](https://github.com/user-attachments/assets/8435ed22-7506-4748-8cae-f0e40b8d2d85)

#### 2. Converse with the portal to and ask any questions on the downloaded episodes.

![Screenshot from 2025-03-16 19-18-40](https://github.com/user-attachments/assets/845669df-c094-47d9-8ba8-c3be964a5a13)

#### 3. Get Summary and important tidbits about the episodes.

![Screenshot from 2025-03-16 19-19-27](https://github.com/user-attachments/assets/c83d48c4-e70c-4604-a3a1-941d19e8121f)

#### 4. Listen to tidbits or the summary of the episodes.

![Screenshot from 2025-03-16 19-20-57](https://github.com/user-attachments/assets/d0035427-eea9-4f0a-987f-123fcaf781c1)

## License

This project is licensed under the MIT License.

## Contributions

Feel free to fork the repository, create feature branches, and submit pull requests!

## Contact

For any questions or support, reach out to [Mahen](https://github.com/kunwarmahen).
