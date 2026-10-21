"""Real file I/O regressions, using only disposable input/cache fixtures."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import folder_paths
from module_fixture import load

cache = load("cache")
mask = load("mask_tools")


class PathSecurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.roots = {name: self.base / name for name in ("input", "cache", "temp", "output", "outside")}
        for root in self.roots.values():
            root.mkdir()
        for method, name in (("get_input_directory", "input"), ("get_output_directory", "output"),
                             ("get_temp_directory", "temp"), ("get_system_user_directory", "cache")):
            p = patch.object(folder_paths, method, return_value=str(self.roots[name]), create=True)
            p.start()
            self.addCleanup(p.stop)
        self.outside = self.roots["outside"] / "sentinel.npz"
        np.savez(self.outside, mask=np.ones((1, 2, 2)))
        self.valid = self.roots["input"] / "valid.npz"
        np.savez(self.valid, mask=np.ones((1, 2, 2)))

    def test_mask_rejects_outside_before_reading(self):
        (self.roots["input"] / "link.npz").symlink_to(self.outside)
        for value in (str(self.outside), "../outside/sentinel.npz", "link.npz"):
            with self.subTest(value=value):
                with patch.object(Path, "open", side_effect=AssertionError("opened before rejection")):
                    with self.assertRaises(ValueError):
                        mask.SavedPersonMask.IS_CHANGED(value)
                with patch.object(mask.np, "load", side_effect=AssertionError("loaded before rejection")):
                    with self.assertRaises(ValueError):
                        mask.SavedPersonMask().load(value, torch.ones(1, 2, 2, 3), 1, 0)

    def test_mask_accepts_valid_file_and_inside_symlink(self):
        (self.roots["input"] / "inside.npz").symlink_to(self.valid)
        digest = hashlib.sha256(self.valid.read_bytes()).hexdigest()
        for value in ("valid.npz", "inside.npz", str(self.valid)):
            self.assertEqual(mask.SavedPersonMask.IS_CHANGED(value), digest)
            expanded, raw, provenance = mask.SavedPersonMask().load(value, torch.ones(1, 2, 2, 3), 1, 0)
            self.assertTrue(torch.equal(expanded, raw))
            self.assertEqual(json.loads(provenance)["sha256"], digest)

    def test_mask_rejects_nonregular_files(self):
        for value in (".", "missing.npz"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mask.SavedPersonMask.IS_CHANGED(value)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO test')
    def test_mask_rejects_fifo_without_opening_it(self):
        os.mkfifo(self.roots['input'] / 'pipe.npz')
        with patch.object(Path, 'open', side_effect=AssertionError('opened FIFO')):
            with self.assertRaises(ValueError):
                mask.SavedPersonMask.IS_CHANGED('pipe.npz')

    def test_cache_roots_reject_traversal_and_symlink_escape(self):
        for name, prefix in (("cache", "cache://"), ("temp", "temp://"), ("output", "")):
            root = self.roots[name]
            (root / "escape").symlink_to(self.roots["outside"], target_is_directory=True)
            (root / "link.npz").symlink_to(self.outside)
            for suffix in ("../outside/sentinel.npz", "escape/sentinel.npz", "escape/new.npz", "link.npz"):
                with self.subTest(root=name, suffix=suffix), self.assertRaises(ValueError):
                    cache.resolve_artifact(prefix + suffix)
            for value in (root / "link.npz", root / "escape/sentinel.npz"):
                for function in (cache.resolve_artifact, cache.artifact_uri, cache.preview_proxy):
                    with self.subTest(function=function.__name__, path=value), self.assertRaises(ValueError):
                        function(str(value))

    def test_cache_write_destination_rejects_outside_symlink(self):
        (self.roots["cache"] / "escape").symlink_to(self.roots["outside"], target_is_directory=True)
        with self.assertRaises(ValueError):
            cache.cache_path("escape", "new.mp4")

    def test_preview_destination_rejects_outside_symlink(self):
        source = self.roots["cache"] / "clip.mp4"
        source.write_bytes(b"valid preview")
        (self.roots["temp"] / "h3_relay").symlink_to(self.roots["outside"], target_is_directory=True)
        with self.assertRaises(ValueError):
            cache.preview_proxy(str(source))
        self.assertFalse((self.roots["outside"] / "cache").exists())

    def test_valid_artifacts_roundtrip_and_preview(self):
        for name in ("cache", "temp", "output"):
            source = self.roots[name] / "clip.mp4"
            source.write_bytes(b"valid preview")
            self.assertEqual(cache.resolve_artifact(cache.artifact_uri(str(source))), str(source))
            proxy, kind = cache.preview_proxy(str(source))
            self.assertEqual(Path(proxy).read_bytes(), b"valid preview")
            self.assertEqual(kind, "output" if name == "output" else "temp")
        alias = self.roots["cache"] / "alias.mp4"
        alias.symlink_to(self.roots["cache"] / "clip.mp4")
        self.assertEqual(cache.resolve_artifact("cache://alias.mp4"), str(alias.resolve()))

    def test_configured_root_may_itself_be_symlink(self):
        alias = self.base / "cache-alias"
        alias.symlink_to(self.roots["cache"], target_is_directory=True)
        with patch.object(folder_paths, "get_system_user_directory", return_value=str(alias)):
            self.assertEqual(cache.cache_path("new.mp4"), str(self.roots["cache"] / "new.mp4"))

    def test_cache_maintenance_does_not_read_or_prune_symlink_targets(self):
        manifest = self.roots["outside"] / "manifest.json"
        manifest.write_text(json.dumps({'revision': 'a' * 32}))
        link = self.roots["cache"] / 'manifest.json'
        link.symlink_to(manifest)
        revision = self.roots["cache"] / ('clip_0001.' + 'b' * 32 + '.json')
        revision.symlink_to(manifest)
        self.assertEqual(cache._protected_revisions(), set())
        self.assertEqual(cache.cache_stats()['files'], 0)
        cache.prune_superseded(keep_per_shot=1)
        self.assertTrue(link.is_symlink())
        self.assertTrue(revision.is_symlink())
        self.assertTrue(manifest.exists())


if __name__ == "__main__":
    unittest.main()
