import os

import requests

# ___ Check if the file exists if not download it to ../data
# If the file is not found, download it from the source URL and save to week_03_rag_memory/data

url ="https://arxiv.org/pdf/1706.03762"


def check_and_download_file(file_path="../data/Attention_is_all_you_need.pdf", url="https://arxiv.org/pdf/1706.03762") -> bool:
    """
    Checks if a file exists in the given file_path. If the file does not exist,
    it downloads the file from the given url and saves it to the file_path.
      
    Args:
        file_path (str): The path to the file to be checked/downloaded.
        url (str): The URL of the file to be downloaded.
   
    Returns:
        bool: True if the file exists or has been downloaded successfully.
              False otherwise.
    """
    status = False
    if not os.path.exists(file_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
       
        print(f"File not found. Downloading from {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
       
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully to {file_path}")
        status = True
    else:
        print("File already exists.")
        status = True
    return status


# check is qdrant db is running on localhost:6333
def check_qdrant_status():
    """
    Check if Qdrant is running.
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        response = requests.get(f"{qdrant_url}/collections")
        if response.status_code == 200:
            print("Qdrant is running.")
            return True
        else:
            print("Qdrant is not running.")
            return False
    except requests.ConnectionError:
        print("Qdrant is not running.")
        return False
    

if __name__ == "__main__":
    file_path = "../data/Attention_is_all_you_need.pdf"
    if check_and_download_file(file_path, url):
        print("File exists or has been downloaded successfully.")
    else:
        print("File not found and download failed.")
    if check_qdrant_status():
        print("Qdrant is running.")
    else:
        print("Qdrant is not running.")
