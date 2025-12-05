import time

from requests.exceptions import (
    ConnectionError as ReqConnectionError,
    RequestException,
    Timeout,
)

from mycelia.shared.client import submit_model
from mycelia.shared.app_logging import configure_logging, structlog
from mycelia.shared.chain import MinerChainCommit, commit_status
from mycelia.shared.checkpoint import delete_old_checkpoints, get_resume_info
from mycelia.shared.config import MinerConfig, parse_args
from mycelia.shared.cycle import (
    search_model_submission_destination,
    setup_chain_worker,
    should_submit_model,
)
from mycelia.shared.model import fetch_model_from_chain

configure_logging()
logger = structlog.get_logger(__name__)

def main(config: MinerConfig) -> None:
    config.write()

    wallet, subtensor = setup_chain_worker(config)

    current_model_version = 0
    current_model_hash = "xxx"
    last_submission_block = 0

    commit_status(
        config,
        wallet,
        subtensor,
        MinerChainCommit(
            expert_group=1,
        ),
    )

    while True:
        try:
            # --------- DOWNLOAD PHASE ---------
            fetch_model_from_chain(current_model_version, current_model_hash, config)
            delete_old_checkpoints(config.ckpt.validator_checkpoint_path, config.ckpt.checkpoint_topk)

            # --------- SUBMISSION PHASE ---------
            _, _, latest_checkpoint_path = get_resume_info(rank=0, config=config)

            start_submit = False
            while not start_submit:
                start_submit, block_till = should_submit_model(config, subtensor, last_submission_block)
                if block_till > 0:
                    time.sleep(block_till * 12)

            destination_axon = search_model_submission_destination(wallet=wallet, config=config, subtensor=subtensor)

            submit_model(
                url=f"http://{destination_axon.ip}:{destination_axon.port}/submit-checkpoint",
                token="",
                my_hotkey=wallet.hotkey,  # type: ignore
                target_hotkey_ss58=destination_axon.hotkey,
                block=subtensor.block,
                model_path=f"{latest_checkpoint_path}/model.pt",
            )

            last_submission_block = subtensor.block

        except (Timeout, ReqConnectionError) as e:
            logger.warning("Network issue: %s", e)
        except RequestException as e:
            logger.warning("HTTP error: %s", e)
        except Exception as e:
            logger.exception("Unexpected error in loop: %s", e)

        time.sleep(60)


if __name__ == "__main__":
    args = parse_args()

    if args.path:
        config = MinerConfig.from_path(args.path)
    else:
        config = MinerConfig()

    main(config)
