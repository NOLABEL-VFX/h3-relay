import unittest
from types import SimpleNamespace
from module_fixture import load

staged_clip = load("text_memory").staged_clip
sampling_workspace = load("text_memory").sampling_workspace


class TextMemoryTests(unittest.TestCase):
    def test_sampling_estimate_is_scoped_and_restored_on_failure(self):
        native = lambda shape, cond: 8 * 2**30
        base = SimpleNamespace(memory_required=native)
        class Model:
            model = base
            def clone(self): return Model()
            def get_model_object(self, name): return getattr(self.model, name)
            def add_wrapper(self, kind, fn): self.wrapper = fn
        releases = []
        management = SimpleNamespace(vram_state="normal", VRAMState=SimpleNamespace(HIGH_VRAM="high"),
                                     unload_all_models=lambda: releases.append(True))
        result = sampling_workspace(Model(), 17., management, "outer")
        self.assertEqual(result.wrapper(lambda: base.memory_required([], {})), 17 * 2**30)
        self.assertIs(base.memory_required, native)
        def fail(): raise ValueError("sampler failure")
        with self.assertRaisesRegex(ValueError, "sampler failure"): result.wrapper(fail)
        self.assertIs(base.memory_required, native)
        self.assertEqual(releases, [True, True, True])

    def test_native_result_and_failure_both_release_without_modifying_input(self):
        events = []
        class Clip:
            def clone(self): return Clip()
            def encode_from_tokens_scheduled(self, tokens, **kwargs):
                events.append((tokens, kwargs))
                if tokens == "fail": raise ValueError("native failure")
                return tokens
        original = Clip()
        management = SimpleNamespace(vram_state="normal", VRAMState=SimpleNamespace(HIGH_VRAM="high"),
                                     unload_all_models=lambda: events.append("release"))
        clone = staged_clip(original, management)
        self.assertIsNot(clone, original)
        self.assertEqual(clone.encode_from_tokens_scheduled("tokens", custom=7), "tokens")
        self.assertEqual(events, ["release", ("tokens", {"custom": 7}), "release"])
        events.clear()
        with self.assertRaisesRegex(ValueError, "native failure"):
            clone.encode_from_tokens_scheduled("fail")
        self.assertEqual(events, ["release", ("fail", {}), "release"])
        events.clear()
        original.encode_from_tokens_scheduled("plain")
        self.assertEqual(events, [("plain", {})])
        management.vram_state = "high"
        self.assertIs(staged_clip(original, management), original)


if __name__ == "__main__": unittest.main()
