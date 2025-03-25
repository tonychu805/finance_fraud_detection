"""
Script to deploy models and create Spaces on Hugging Face Hub.
"""
import os
from huggingface_hub import (
    create_repo,
    HfApi,
    Repository,
    upload_file
)

def deploy_to_hub():
    # Initialize Hugging Face API
    api = HfApi()
    
    # Configuration
    REPO_NAME = "finance-fraud-detection"
    MODEL_PATH = "models/fraud_detector.pkl"
    SPACE_NAME = "finance-fraud-detection-demo"
    
    try:
        # Create or get model repository
        repo_url = create_repo(
            repo_id=REPO_NAME,
            exist_ok=True,
            token=os.environ["HF_TOKEN"]
        )
        
        # Clone repository
        repo = Repository(
            local_dir="hf_repo",
            clone_from=repo_url,
            token=os.environ["HF_TOKEN"]
        )
        
        # Upload model file
        if os.path.exists(MODEL_PATH):
            upload_file(
                path_or_fileobj=MODEL_PATH,
                path_in_repo="model/fraud_detector.pkl",
                repo_id=REPO_NAME,
                token=os.environ["HF_TOKEN"]
            )
        
        # Create or update Spaces files
        create_repo(
            repo_id=SPACE_NAME,
            exist_ok=True,
            token=os.environ["HF_TOKEN"],
            repo_type="space",
            space_sdk="gradio"
        )
        
        # Upload Space files
        space_files = [
            "app.py",
            "requirements.txt",
            "README.md"
        ]
        
        for file in space_files:
            if os.path.exists(f"spaces/{file}"):
                upload_file(
                    path_or_fileobj=f"spaces/{file}",
                    path_in_repo=file,
                    repo_id=SPACE_NAME,
                    token=os.environ["HF_TOKEN"],
                    repo_type="space"
                )
        
        print("✅ Deployment successful!")
        print(f"🤗 Model: https://huggingface.co/{REPO_NAME}")
        print(f"🚀 Demo: https://huggingface.co/spaces/{SPACE_NAME}")
        
    except Exception as e:
        print(f"❌ Error during deployment: {str(e)}")
        raise e

if __name__ == "__main__":
    deploy_to_hub() 