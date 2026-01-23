from huggingface_hub import login, create_repo, upload_folder
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("uploadHF")

login(add_to_git_credential=True)
repo_id = "exdsgift/NerGuard"
try:
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    logger.info(f"uploaded files in: {repo_id} ")
except Exception as e:
    logger.debug(f"Error while creating repo: {e}")
    logger.info("Continue uploading ...")

logger.info("Uploading model ...")
upload_folder(
    folder_path="models/quantized_model",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload mdeberta-pii-safe model"
)

logger.info("- Model uploaded succesfully!")
