"""Sanity checks for llama.cpp MoE expert-offload argument selection.

Imports chat_server (the launch path) and checks the pure flag builder
against synthetic GGUF headers. No GPU and no llama-server binary required.
"""

import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Shared"))
import chat_server as cs  # noqa: E402


B9444 = {
    "fit": True,
    "n_cpu_moe": True,
    "cpu_moe": True,
    "gpu_layers_flag": "--gpu-layers",
    "probed": True,
}

NO_FIT = {
    "fit": False,
    "n_cpu_moe": True,
    "cpu_moe": True,
    "gpu_layers_flag": "--gpu-layers",
    "probed": True,
}


def _gguf_string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key, value):
    return _gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value)


def _kv_u32(key, value):
    return _gguf_string(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def _kv_str_array(key, items):
    body = struct.pack("<I", 9) + struct.pack("<IQ", 8, len(items))
    for item in items:
        body += _gguf_string(item)
    return _gguf_string(key) + body


def build_gguf(parts):
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(parts)) + b"".join(parts)


def write_gguf(directory, name, parts):
    path = os.path.join(directory, name)
    with open(path, "wb") as handle:
        handle.write(build_gguf(parts))
    return path


class MoeOffloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _moe_file(self, experts=128, layers=48, arch="qwen3moe"):
        return write_gguf(self.tmp, "Qwen3-30B-A3B-Q4.gguf", [
            _kv_string("general.architecture", arch),
            _kv_u32("%s.expert_count" % arch, experts),
            _kv_u32("%s.expert_used_count" % arch, 8),
            _kv_u32("%s.block_count" % arch, layers),
            _kv_str_array("tokenizer.ggml.tokens", ["hello", "world"]),
            _kv_string("tokenizer.chat_template", "template <tool_call>"),
        ])

    def _dense_file(self):
        return write_gguf(self.tmp, "gemma-2-2b-q4.gguf", [
            _kv_string("general.architecture", "gemma"),
            _kv_u32("gemma.block_count", 26),
            _kv_u32("gemma.expert_count", 0),
            _kv_string("tokenizer.chat_template", "dense-template"),
        ])

    def _plan(self, path, mode="auto", layers=28, discrete=True, caps=None, vram=None, file_bytes=None):
        return cs._llama_moe_offload_plan(
            path,
            settings={"llamaMoeOffload": mode, "llamaMoeOffloadLayers": layers},
            discrete_gpu=discrete,
            capabilities=B9444 if caps is None else caps,
            vram_bytes=0 if vram is None else vram,
            file_bytes=file_bytes,
        )

    def test_module_exports_launch_helpers(self):
        self.assertTrue(callable(cs._start_llama))
        self.assertTrue(callable(cs._llama_moe_offload_plan))
        self.assertTrue(callable(cs._normalize_moe_settings))

    def test_chat_template_reader_still_works(self):
        path = self._moe_file()
        self.assertEqual(cs._gguf_chat_template(path), "template <tool_call>")

    def test_metadata_distinguishes_expert_count_from_used_count(self):
        meta = cs._gguf_moe_metadata(self._moe_file())
        self.assertEqual(meta["architecture"], "qwen3moe")
        self.assertEqual(meta["expert_count"], 128)
        self.assertEqual(meta["expert_used_count"], 8)
        self.assertEqual(meta["block_count"], 48)

    def test_filename_fallback_when_metadata_has_no_expert_count(self):
        path = write_gguf(self.tmp, "gpt-oss-20b-mxfp4.gguf", [
            _kv_string("general.architecture", "gpt-oss"),
            _kv_u32("gpt-oss.block_count", 24),
        ])
        facts = cs._moe_model_facts(path)
        self.assertTrue(facts["is_moe"])
        self.assertEqual(facts["source"], "filename")
        self.assertEqual(facts["block_count"], 24)

    def test_explicit_zero_experts_beats_a_moe_filename(self):
        path = write_gguf(self.tmp, "not-really-moe-q4.gguf", [
            _kv_string("general.architecture", "llama"),
            _kv_u32("llama.expert_count", 0),
            _kv_u32("llama.block_count", 32),
        ])
        self.assertFalse(cs._moe_model_facts(path)["is_moe"])

    def test_auto_moe_on_discrete_gpu_uses_fit_and_not_ngl(self):
        args, reason = self._plan(self._moe_file(), mode="auto")
        self.assertEqual(args, ["--fit", "on"])
        self.assertNotIn("--n-cpu-moe", args)
        self.assertNotIn("--gpu-layers", args)
        self.assertIn("--fit", reason)

    def test_auto_dense_and_non_discrete_add_nothing(self):
        dense_args, _reason = self._plan(self._dense_file(), mode="auto", discrete=True)
        self.assertEqual(dense_args, [])
        integrated_args, reason = self._plan(self._moe_file(), mode="manual", layers=28, discrete=False)
        self.assertEqual(integrated_args, [])
        self.assertIn("launch flags unchanged", reason)
        off_dense, _reason = self._plan(self._dense_file(), mode="off", discrete=True)
        self.assertEqual(off_dense, [])

    def test_off_disables_fit_only_for_discrete_moe(self):
        args, _reason = self._plan(self._moe_file(), mode="off")
        self.assertEqual(args, ["--fit", "off"])
        untouched, _reason = self._plan(self._moe_file(), mode="off", discrete=False)
        self.assertEqual(untouched, [])

    def test_manual_partial_and_full_layer_counts(self):
        partial, _reason = self._plan(self._moe_file(), mode="manual", layers=28)
        self.assertEqual(partial, ["--fit", "off", "--gpu-layers", "999", "--n-cpu-moe", "28"])
        full, reason = self._plan(self._moe_file(), mode="manual", layers=48)
        self.assertEqual(full, ["--fit", "off", "--gpu-layers", "999", "--cpu-moe"])
        self.assertIn("--cpu-moe", reason)
        zero, _reason = self._plan(self._moe_file(), mode="manual", layers=0)
        self.assertEqual(zero, ["--fit", "off"])

    def test_auto_without_fit_offloads_only_when_file_exceeds_vram(self):
        big, _reason = self._plan(
            self._moe_file(),
            mode="auto",
            caps=NO_FIT,
            vram=8 * 1024 ** 3,
            file_bytes=17 * 1024 ** 3,
        )
        self.assertEqual(big, ["--gpu-layers", "999", "--cpu-moe"])
        fits, _reason = self._plan(
            self._moe_file(),
            mode="auto",
            caps=NO_FIT,
            vram=24 * 1024 ** 3,
            file_bytes=4 * 1024 ** 3,
        )
        self.assertEqual(fits, [])
        unknown, _reason = self._plan(
            self._moe_file(),
            mode="auto",
            caps=NO_FIT,
            vram=0,
            file_bytes=17 * 1024 ** 3,
        )
        self.assertEqual(unknown, [])

    def test_help_flag_boundaries(self):
        text = "--n-cpu-moe N\n--fit-target\n--fit-ctx\n-fit, --fit [on|off]\n--cpu-moe\n--gpu-layers, --n-gpu-layers"
        self.assertFalse(cs._help_has_flag("--n-cpu-moe N", "--cpu-moe"))
        self.assertTrue(cs._help_has_flag(text, "--cpu-moe"))
        self.assertFalse(cs._help_has_flag("--fit-target 1024", "--fit"))
        self.assertTrue(cs._help_has_flag(text, "--fit"))
        self.assertTrue(cs._help_has_flag(text, "--n-cpu-moe"))

    def test_device_list_ignores_integrated_and_sums_discrete_free_vram(self):
        text = "\n".join([
            "Available devices:",
            "  CUDA0: NVIDIA GeForce RTX 3070 Laptop GPU (8192 MiB, 7100 MiB free)",
            "  Vulkan1: Intel(R) UHD Graphics (2048 MiB, 900 MiB free)",
        ])
        lines = cs._device_list_lines(text)
        self.assertTrue(any(cs._line_is_discrete_device(line) for line in lines))
        self.assertFalse(cs._line_is_discrete_device(
            "Vulkan0: Intel(R) UHD Graphics 620 (1024 MiB, 800 MiB free)"
        ))
        self.assertFalse(cs._line_is_discrete_device(
            "ROCm0: AMD Radeon Graphics (4096 MiB, 3000 MiB free)"
        ))
        self.assertTrue(cs._line_is_discrete_device(
            "Vulkan0: AMD Radeon RX 6800 (16384 MiB, 15000 MiB free)"
        ))
        self.assertTrue(cs._vulkan_summary_has_discrete(
            "deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
        ))
        self.assertEqual(cs._vram_bytes_from_device_lines(lines), 7100 * 1024 * 1024)

    def test_filename_patterns(self):
        self.assertTrue(cs._filename_looks_like_moe("Qwen3-30B-A3B-Q4_K_M.gguf"))
        self.assertTrue(cs._filename_looks_like_moe("Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"))
        self.assertTrue(cs._filename_looks_like_moe("gpt-oss-20b.gguf"))
        self.assertTrue(cs._filename_looks_like_moe("mixtral-8x7b-instruct.gguf"))
        self.assertTrue(cs._filename_looks_like_moe("DeepSeek-MoE-q4.gguf"))
        self.assertFalse(cs._filename_looks_like_moe("gemma-2-2b.gguf"))
        self.assertFalse(cs._filename_looks_like_moe("qwen2.5-vl-3b.gguf"))
        self.assertFalse(cs._filename_looks_like_moe("nemo-mix-12b.gguf"))

    def test_settings_normalization(self):
        path = os.path.join(self.tmp, "settings.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"llamaMoeOffload": " MANUAL ", "llamaMoeOffloadLayers": "400"}')
        previous = cs.SETTINGS_FILE
        cs.SETTINGS_FILE = path
        try:
            loaded = cs._load_settings_file()
        finally:
            cs.SETTINGS_FILE = previous
        self.assertEqual(loaded["llamaMoeOffload"], "manual")
        self.assertEqual(loaded["llamaMoeOffloadLayers"], 256)
        self.assertEqual(loaded["logMode"], cs.DEFAULT_LOG_MODE)

    def test_missing_settings_file_defaults_to_auto(self):
        previous = cs.SETTINGS_FILE
        cs.SETTINGS_FILE = os.path.join(self.tmp, "absent.json")
        try:
            loaded = cs._load_settings_file()
        finally:
            cs.SETTINGS_FILE = previous
        self.assertEqual(loaded["llamaMoeOffload"], "auto")
        self.assertEqual(loaded["llamaMoeOffloadLayers"], 28)


if __name__ == "__main__":
    unittest.main()
