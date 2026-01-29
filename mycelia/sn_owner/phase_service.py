import bittensor
import hivemind
from mycelia.validator.inter_validator_connection import HotkeyAuthorizer
import uvicorn
from fastapi import FastAPI, HTTPException

from mycelia.shared.config import OwnerConfig, parse_args
from mycelia.sn_owner.cycle import PhaseManager, PhaseResponse
from mycelia.shared.helper import public_multiaddrs

app = FastAPI(title="Phase Service")

@app.get("/get_phase", response_model=PhaseResponse)
async def read_phase():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return phase_manager.get_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/previous_phase_blocks", response_model=dict[str, tuple[int, int]])
async def prev_phase():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return phase_manager.previous_phase_block_ranges()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/blocks_until_next_phase", response_model=dict[str, tuple[int, int, int]])
async def next_phase():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return phase_manager.blocks_until_next_phase()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/get_init_peer_id", response_model=str)
async def get_init_peer_id():
    """
    Returns which phase we're in for the given block height.
    """
    try:
        return init_peer_id
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/")
async def root():
    return {
        "message": "Phase service is running",
        "cycle_length": phase_manager.cycle_length,
        "phases": [{"index": i, "name": p["name"], "length": p["length"]} for i, p in enumerate(phase_manager.phases)],
        "usage": "GET /phase?block_height=123",
    }


if __name__ == "__main__":
    args = parse_args()

    global config
    global phase_manager
    global init_peer_id

    if args.path:
        config = OwnerConfig.from_path(args.path)
    else:
        config = OwnerConfig()

    config.write()
    # bittensor setup 
    subtensor = bittensor.Subtensor(network=config.chain.network)
    wallet = bittensor.Wallet(name=config.chain.coldkey_name, hotkey=config.chain.hotkey_name)

    # DHT setup 
    authorizer = HotkeyAuthorizer(
        my_hotkey=wallet.hotkey,
        max_time_skew_s=30.0,
        subtensor=subtensor,
        config = config,
    )

    dht = hivemind.DHT(
        host_maddrs=["/ip4/0.0.0.0/tcp/34297", "/ip4/0.0.0.0/udp/34297/quic"],
        start=True,
        client_mode = False,
        authorizer=authorizer
    )

    init_peer_id = str(public_multiaddrs(dht.get_visible_maddrs())[0])

    phase_manager = PhaseManager(config, subtensor)
    uvicorn.run(app, host=config.owner.app_ip, port=config.owner.app_port)
