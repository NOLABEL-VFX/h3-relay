"""Timeline, edit-mask, and legacy behavior contracts for reusable H3 nodes."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
# Load these adapters without importing model loaders or installing patches.
PACKAGE = "h3_toolkit_unit_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "h3_relay")]
sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(PACKAGE + ".toolkit", ROOT / "h3_relay/toolkit.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class ToolkitTests(unittest.TestCase):
    def source(self, count):
        return torch.arange(count).float()[:, None, None, None].expand(-1, 32, 32, 3)

    def audio(self, count, rate=240):
        return {"sample_rate": rate, "waveform": torch.arange(round(count*rate/24)).float()[None, None]}

    def test_adaptive_source_and_audio_alignment(self):
        for size in (22, 39, 124):
            for history in (0, 1, 18):
                for count in (1, 21, 22, 23, 39, 40, 124, 144, 240):
                    with self.subTest(size=size, history=history, count=count):
                        source, audio = self.source(count), self.audio(count)
                        entries, _ = m.H3RelayWindowPlan().plan(count, size, history, 100)
                        previous = None
                        for entry in entries:
                            selected = m.H3RelayWindowInputs().select(source, entry, source_audio=audio, previous=previous)
                            start, h = entry["source_start"], entry["history_frames"]
                            begin = start-max(0, h-1)
                            self.assertEqual(selected[5][0, 0, 0, 0].item(), begin)
                            self.assertEqual(selected[6]["waveform"][0, 0, 0].item(), begin*10)
                            self.assertEqual(selected[2]["waveform"].shape[-1], entry["window_frames"]*10)
                            previous, = m.H3RelayAppendWindow().append(selected[0], entry, previous, selected[2])
                        frames, output_audio, fps, report = m.H3RelayFinishWindows().finish(previous, "generated")
                        self.assertTrue(torch.equal(frames, source))
                        self.assertTrue(torch.equal(output_audio["waveform"], audio["waveform"]))
                        self.assertEqual(json.loads(report)["delivered_frames"], count)
                        self.assertEqual(fps, 24)

    def test_person_removal_schedule_retains_tail_overlap(self):
        entries, _ = m.H3RelayWindowPlan().plan(124, 22, 18, 904234, "removal_v1", "increment")
        self.assertEqual([e["source_start"] for e in entries], [0, 21, 42, 63, 84, 102])
        self.assertEqual(entries[-1]["discarded_overlap"], 4)
        self.assertEqual([e["seed"] for e in entries], list(range(904234, 904240)))
        state = None
        for entry in entries:
            values = m.H3RelayWindowInputs().select(self.source(124), entry, False, "generated", state,
                                                  source_audio=self.audio(124))
            state, = m.H3RelayAppendWindow().append(values[0], entry, state, values[2])
        self.assertTrue(torch.equal(state["frames"], self.source(124)))
        self.assertTrue(torch.equal(state["audio"]["waveform"], self.audio(124)["waveform"]))

    def test_segment_excludes_supplied_history_from_delivery(self):
        source, audio = self.source(58), self.audio(58)
        history = source[:18].clone()
        entries, _ = m.H3RelayWindowPlan().plan(58, 124, 18, 9, "segment", initial_history=history)
        self.assertEqual(entries[0]["source_start"], 17)
        selected = m.H3RelayWindowInputs().select(source, entries[0], source_audio=audio, initial_history=history)
        self.assertEqual(selected[5][0, 0, 0, 0].item(), 0)
        state, = m.H3RelayAppendWindow().append(selected[0], entries[0], initial_history=history)
        frames, out_audio, _, report = m.H3RelayFinishWindows().finish(state, source_audio=audio)
        self.assertTrue(torch.equal(frames, source[18:]))
        self.assertTrue(torch.equal(out_audio["waveform"], audio["waveform"][..., 180:]))
        self.assertEqual(json.loads(report)["history_frames"], 18)

    def test_mask_temporal_union_does_not_lose_short_events(self):
        mask = torch.zeros(22, 64, 64)
        mask[3, 7, 9] = 1
        pooled = m.pool_edit_mask(mask, (1, 24, 7, 4, 4))
        self.assertEqual(pooled.shape, (1, 1, 7, 4, 4))
        self.assertEqual(pooled.sum().item(), 4)
        self.assertTrue(torch.all(pooled[0, 0, 1, :2, :2] == 1))

    def test_seed_overrides_and_invalid_configuration(self):
        entries, _ = m.H3RelayWindowPlan().plan(144, 124, 18, 9, seed_overrides='{"1":42}')
        self.assertEqual([e["seed"] for e in entries], [9, 42])
        for value in ('[]', '{"7":42}', '{"0":-1}', '{"0":true}'):
            with self.assertRaises(ValueError):
                m.H3RelayWindowPlan().plan(144, 124, 18, 9, seed_overrides=value)

    def test_assembly_rejects_gap_and_incomplete_output(self):
        entry = m.H3RelayWindowPlan().plan(44, 22, 18, 9)[0][1]
        with self.assertRaisesRegex(ValueError, "gap or duplicate"):
            m.H3RelayAppendWindow().append(self.source(22), entry)
        with self.assertRaisesRegex(ValueError, "exact source timeline"):
            m.H3RelayFinishWindows().finish({"contract": m.CONTRACT, "frames": self.source(22), "source_frames": 44})


if __name__ == "__main__":
    unittest.main()
