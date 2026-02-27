import os
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import delete_repo, create_repo, upload_folder

# pushing outside of training ensures we have control over whether
# we want to overwrite the existing max_checkpoints_to_keep, store
# the current checkpoints (on the Hub) elsewhere before overwriting
# the HF repo, or even add the local checkpoints to the Hub without 
# deleting old ones if we skip calling the delete_repo function
timestamp = "20260211_180749"
config_and_log_dir = f"./configs_and_logs/{timestamp}"
hf_user = os.environ.get("hf_user")
hf_token = os.environ.get("hf_token")
hub_repo_id = f"{hf_user}/nanogpt_{timestamp}"
checkpoint_dir = f"./checkpoints/{timestamp}"

def push_to_hub():
    try:
        upload_folder(
        repo_id=hub_repo_id,
        folder_path=checkpoint_dir,
        commit_message=f"checkpoints",
        token=hf_token,
        repo_type="model",
        )
        print("successfully pushed checkpoint dir to HF")
        upload_folder(
            repo_id=hub_repo_id,
            folder_path=config_and_log_dir,
            commit_message=f"config and logs",
            token=hf_token,
            repo_type="model",
        )
        print("successfully pushed config and logs dir to HF")
        return True
    except Exception as e:
        print(f"failed to push config and logs dir to HF: {e}")
        return False
    
try:
    delete_repo(repo_id=hub_repo_id, token=hf_token, repo_type="model")
    print(f"repository deleted: {hub_repo_id}")
except HfHubHTTPError as e:
    if e.response.status_code == 404:
        print(f"repository {hub_repo_id} did not exist, skipping deletion.")
    else:
        raise
create_repo(repo_id=hub_repo_id, exist_ok=True, token=hf_token, repo_type="model")
print(f"repository created: {hub_repo_id}")
push_to_hub()
print(f"repository pushed to HF repo id: {hub_repo_id}")