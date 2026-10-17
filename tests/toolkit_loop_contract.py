"""Execute native ComfyUI loops with real H3 timeline adapters, on CPU."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--comfyui-root", required=True)
args = parser.parse_args()
sys.path[:0] = [args.comfyui_root, str(Path(__file__).resolve().parents[1])]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import nodes
import execution
from server import PromptServer
from app.assets.manager import NoAssets
from comfy.cli_args import args as comfy_args


class Fixture:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"count": ("INT", {"default": 44})}}
    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    FUNCTION = "get"
    def get(self, count):
        frames = torch.arange(count).float()[:, None, None, None].expand(-1, 32, 32, 3)
        audio = {"waveform": torch.arange(count*10).float()[None, None], "sample_rate": 240}
        return frames, audio, count


class AssertOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "audio": ("AUDIO",), "count": ("INT",)}}
    RETURN_TYPES = ()
    FUNCTION = "check"
    OUTPUT_NODE = True
    def check(self, frames, audio, count):
        assert frames[:, 0, 0, 0].tolist() == list(range(count))
        assert audio["waveform"].flatten().tolist() == list(range(count*10))
        return {"ui": {"text": [f"{count} source frames and {count*10} audio samples preserved"]}}


async def main():
    assets = NoAssets(comfy_args)
    server = PromptServer(asyncio.get_running_loop(), assets)
    execution.PromptQueue(server)
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    from h3_relay.toolkit import NODE_CLASS_MAPPINGS
    nodes.NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
    nodes.NODE_CLASS_MAPPINGS.update(ToolkitFixture=Fixture, ToolkitAssert=AssertOutput)
    graph = {
        "source": {"class_type": "ToolkitFixture", "inputs": {"count": 44}},
        "plan": {"class_type": "H3RelayWindowPlan", "inputs": {"source_frames": ["source", 2],
            "window_frames": "22", "history_frames": "18", "seed": 9, "schedule": "adaptive", "seed_mode": "fixed"}},
        "start": {"class_type": "StartLoop", "inputs": {"mode": "List", "mode.list": ["plan", 0], "cache_iterations": True}},
        "window": {"class_type": "H3RelayWindowInputs", "inputs": {"source": ["source", 0],
            "source_audio": ["source", 1], "window": ["start", 3], "previous": ["start", 4],
            "reference_history": True, "history_audio_mode": "source"}},
        "append": {"class_type": "H3RelayAppendWindow", "inputs": {"frames": ["window", 0],
            "audio": ["window", 2], "window": ["start", 3], "previous": ["start", 4]}},
        "end": {"class_type": "EndLoop", "inputs": {"output_value": ["append", 0],
            "next_iteration_value": ["append", 0], "accumulate": False}},
        "finish": {"class_type": "H3RelayFinishWindows", "inputs": {"state": ["end", 0], "audio_mode": "generated"}},
        "assert": {"class_type": "ToolkitAssert", "inputs": {"frames": ["finish", 0], "audio": ["finish", 1], "count": 44}},
    }
    prompt_id = str(uuid.uuid4())
    valid = await execution.validate_prompt(prompt_id, graph, None)
    if not valid[0]:
        raise AssertionError(json.dumps(valid, default=str))
    executor = execution.PromptExecutor(server, cache_type=execution.CacheType.CLASSIC,
        cache_args={"lru": 0, "ram": 0, "ram_inactive": 0}, asset_manager=assets)
    await executor.execute_async(graph, prompt_id, {}, valid[2])
    if not executor.success:
        raise AssertionError(executor.status_messages)
    print("TOOLKIT_NATIVE_LOOP_PASS", json.dumps(executor.history_result, default=str))
    if server.client_session is not None:
        await server.client_session.close()


asyncio.run(main())
