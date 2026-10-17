"""Execute native ComfyUI loops with real H3 timeline adapters, on CPU."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import uuid
import tempfile

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


class CountGeneration:
    calls = 0
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "audio": ("AUDIO",)}}
    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "generate"
    def generate(self, frames, audio):
        type(self).calls += 1
        return frames, audio


class AssertBlackOutput(AssertOutput):
    def check(self, frames, audio, count):
        assert len(frames) == count and torch.count_nonzero(frames) == 0
        assert audio["waveform"].flatten().tolist() == list(range(count*10))
        return {"ui": {"text": ["Prepared source and audio preserved"]}}


async def main():
    assets = NoAssets(comfy_args)
    server = PromptServer(asyncio.get_running_loop(), assets)
    execution.PromptQueue(server)
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    from h3_relay.toolkit import NODE_CLASS_MAPPINGS
    from h3_relay.window_review import NODE_CLASS_MAPPINGS as review_nodes
    from h3_relay import removal_cache, removal_previews
    nodes.NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS)
    nodes.NODE_CLASS_MAPPINGS.update(review_nodes)
    nodes.NODE_CLASS_MAPPINGS.update(ToolkitFixture=Fixture, ToolkitAssert=AssertOutput,
        ToolkitGeneration=CountGeneration, ToolkitAssertBlack=AssertBlackOutput)
    graph = {
        "source": {"class_type": "ToolkitFixture", "inputs": {"count": 44}},
        "plan": {"class_type": "H3RelayWindowPlan", "inputs": {"source_frames": ["source", 2],
            "window_frames": "22", "history_frames": "18", "seed": 9, "schedule": "adaptive", "seed_mode": "fixed"}},
        "start": {"class_type": "StartLoop", "inputs": {"mode": "simple", "mode.num_iterations": ["plan", 2], "cache_iterations": False}},
        "window": {"class_type": "H3RelayWindowInputs", "inputs": {"source": ["source", 0],
            "source_audio": ["source", 1], "window": ["plan", 3], "iteration_index": ["start", 0], "previous": ["start", 4],
            "reference_history": True, "history_audio_mode": "source", "fps": 24.}},
        "append": {"class_type": "H3RelayAppendWindow", "inputs": {"frames": ["window", 0],
            "audio": ["window", 2], "window": ["window", 12], "previous": ["start", 4]}},
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
    # New executor simulates an empty in-memory cache after restart. The exact
    # locked prefix must be restored from disk without evaluating generation.
    import folder_paths
    with tempfile.TemporaryDirectory(prefix="h3-toolkit-review-") as temporary:
        folder_paths.set_output_directory(temporary)
        graph["anchor"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["source", 0], "batch_index": 0, "length": 1}}
        graph["review"] = {"class_type": "H3RelayReviewWindows", "inputs": {"windows": ["plan", 0],
            "source": ["source", 0], "anchor": ["anchor", 0], "window_controls": "{}"}}
        graph["start"]["inputs"].update({"mode.num_iterations": ["review", 2], "initial_iteration_value": ["review", 1]})
        graph["window"]["inputs"]["window"] = ["review", 3]
        graph["generate"] = {"class_type": "ToolkitGeneration", "inputs": {"frames": ["window", 0], "audio": ["window", 2]}}
        graph["append"]["inputs"].update(frames=["generate", 0], audio=["generate", 1])

        async def render():
            prompt_id = str(uuid.uuid4())
            valid = await execution.validate_prompt(prompt_id, graph, None)
            assert valid[0], valid
            executor = execution.PromptExecutor(server, cache_type=execution.CacheType.CLASSIC,
                cache_args={"lru": 0, "ram": 0, "ram_inactive": 0}, asset_manager=assets)
            await executor.execute_async(graph, prompt_id, {}, valid[2])
            assert executor.success, executor.status_messages
            return removal_previews.latest_run("", "review")

        original = await render()
        assert CountGeneration.calls == 3, CountGeneration.calls
        prefix = original["segments"][:-1]
        controls = {"config_key": original["config_key"], "windows": {
            str(s["index"]): {"seed": s["seed"], "locked": True, "record_id": s["record_id"]} for s in prefix}}
        controls["windows"]["2"] = {"seed": "42", "locked": False}
        graph["review"]["inputs"]["window_controls"] = json.dumps(controls)
        rerolled = await render()
        assert CountGeneration.calls == 4, CountGeneration.calls
        assert [s["record_id"] for s in rerolled["segments"][:2]] == [s["record_id"] for s in prefix]
        assert all(s["reused"] for s in rerolled["segments"][:2])
        assert rerolled["segments"][2]["seed"] == "42"
        # A changed generation parameter invalidates the complete prefix.
        graph["plan"]["inputs"]["seed"] = 100
        changed = await render()
        assert CountGeneration.calls == 7, CountGeneration.calls
        assert changed["controls_reset"] and not any(s["reused"] for s in changed["segments"])
        print("TOOLKIT_DURABLE_REROLL_PASS", json.dumps({"original_windows": 3, "reroll_generations": 1,
            "preserved_prefix": 2, "changed_config_regenerated": 3}))

        class InstrumentedEmptyImage(nodes.NODE_CLASS_MAPPINGS["EmptyImage"]):
            source_calls = 0
            def generate(self, width, height, batch_size=1, color=0):
                if batch_size == 44:
                    type(self).source_calls += 1
                return super().generate(width, height, batch_size, color)

        nodes.NODE_CLASS_MAPPINGS["EmptyImage"] = InstrumentedEmptyImage
        graph["prepared"] = {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "batch_size": 44, "color": 0}}
        graph["anchor"] = {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "batch_size": 1, "color": 0}}
        graph["review"]["inputs"].update(source=["prepared", 0], window_controls="{}")
        graph["window"]["inputs"]["source"] = ["review", 4]
        graph["assert"]["class_type"] = "ToolkitAssertBlack"
        cached = await render()
        assert InstrumentedEmptyImage.source_calls == 1
        controls = {"config_key": cached["config_key"], "windows": {
            str(s["index"]): {"seed": s["seed"], "locked": True, "record_id": s["record_id"]}
            for s in cached["segments"][:2]}}
        controls["windows"]["2"] = {"seed": "43", "locked": False}
        graph["review"]["inputs"]["window_controls"] = json.dumps(controls)
        await render()
        assert InstrumentedEmptyImage.source_calls == 1, "Prepared source should not re-execute after an executor restart"
        print("TOOLKIT_PREPARED_SOURCE_CACHE_PASS")
    if server.client_session is not None:
        await server.client_session.close()


asyncio.run(main())
