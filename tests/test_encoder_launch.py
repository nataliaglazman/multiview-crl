"""Encoder-only experiments through scripts/launch.py: YAML resolution, the trainer's own checks, scripts."""

import argparse
import contextlib
import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import launch
from training import encoder_args

ROOT = Path(__file__).resolve().parents[1]
ENCODER = ROOT / "experiments" / "encoder"
CLUSTERS = ("runai", "slurm", "slurm_bio")
PATH_KEYS = {"dataroot", "labels_path", "masks_dir", "cache_dir"}
SYNTHETIC = ("conv", "conv_mlp", "resnet18", "resnet_groupnorm", "resnet_stride8", "conv_mlp_patch", "conv_mlp_gumbel")


def resolve(name, cluster="runai", **overrides):
    return launch.resolve_config(ENCODER / f"{name}.yaml", cluster, overrides)


def parsed(config):
    return vars(encoder_args.parse_args(launch.encoder_cli_args(config)))


def exits(test, code, function, *args):
    """Run ``function`` expecting ``sys.exit(code)``; returns what it printed to stderr."""
    messages = io.StringIO()
    with contextlib.redirect_stderr(messages), contextlib.redirect_stdout(io.StringIO()):
        with test.assertRaises(SystemExit) as raised:
            function(*args)
    test.assertEqual(raised.exception.code, code)
    return messages.getvalue()


def trainer_command(tokens):
    """The trainer's arguments in a command, after ``-m training.main_conv_synthetic``."""
    return tokens[tokens.index(launch.ENCODER_ENTRYPOINT) + 1 :]


class ExperimentTests(unittest.TestCase):
    def test_every_experiment_passes_the_trainers_parser_on_every_cluster(self):
        experiments = launch.find_experiment_yamls(ENCODER)
        self.assertEqual(
            {path.stem for path in experiments},
            {
                *SYNTHETIC,
                "conv_mlp_patch_separate_spatial",
                "conv_mlp_patch_lesion_corrected",
                "adni",
                "adni_gumbel",
                "adni_patch",
            },
        )
        for path in experiments:
            for cluster in (*CLUSTERS, "local"):
                if path.stem.startswith("adni") and cluster == "local":
                    continue
                with self.subTest(experiment=path.stem, cluster=cluster):
                    config = launch.resolve_config(path, cluster, {})
                    self.assertTrue(launch.is_encoder(config))
                    self.assertEqual(parsed(config)["model_id"], config["tag"])
                    self.assertNotIn("{", config["tag"])

    def test_synthetic_recipe_is_the_paired_comparison_recipe(self):
        recipe = json.loads((ROOT / "experiments" / "encoder_comparison.json").read_text())
        for name in ("conv", "resnet18"):
            with self.subTest(name=name):
                args = parsed(resolve(name))
                for key, value in recipe["shared"].items():
                    self.assertEqual(args[key], value, key)
                self.assertEqual((args["data_seed"], args["seed"]), (recipe["data_seed"], recipe["seeds"][0]))

    def test_variants_differ_from_their_reference_only_by_the_intervention(self):
        for reference, variant, keys in (
            ("conv", "conv_mlp", {"conv_readout"}),
            ("resnet18", "resnet_groupnorm", {"resnet_norm"}),
            ("resnet18", "resnet_stride8", {"resnet_output_stride"}),
            # train_patch_grid is 8^3 by default, so only the weight and the probes change.
            ("conv_mlp", "conv_mlp_patch", {"patch_loss_weight", "spatial_recovery_eval"}),
            ("conv_mlp_patch", "conv_mlp_patch_separate_spatial", {"separate_spatial_readout"}),
            ("conv_mlp_patch", "conv_mlp_patch_lesion_corrected", {"synthetic_lesion_intensity", "train_steps"}),
            ("conv_mlp", "conv_mlp_gumbel", {"mask_mode", "mask_lr_scale"}),
            ("adni", "adni_gumbel", {"mask_mode", "mask_lr_scale"}),
            ("adni", "adni_patch", {"patch_loss_weight", "train_patch_grid", "patch_foreground_mask"}),
        ):
            with self.subTest(variant=variant):
                a, b = parsed(resolve(reference)), parsed(resolve(variant))
                changed = {key for key in a if a[key] != b[key]} - {"model_id", "spatial_recovery_grids"}
                self.assertEqual(changed, keys)

    def test_a_seed_override_renames_the_run_and_keeps_the_paired_data(self):
        args = parsed(resolve("conv_mlp", seed=142))
        self.assertEqual(args["model_id"], "conv_mlp_s142")
        self.assertEqual((args["data_seed"], args["model_seed"], args["loader_seed"]), (42, 142, 10142))
        self.assertEqual(parsed(resolve("conv_mlp", seed=142, tag="mine"))["model_id"], "mine")
        exits(self, 1, lambda: resolve("conv_mlp", "runai", tag="run_{no_such_key}"))

    def test_cluster_paths_reach_real_data_only(self):
        for name in SYNTHETIC:
            runs = [parsed(resolve(name, cluster)) for cluster in CLUSTERS]
            with self.subTest(name=name):
                self.assertTrue(all(run == runs[0] for run in runs))
                self.assertTrue(all(runs[0][key] is None for key in PATH_KEYS))
        runs = {cluster: parsed(resolve("adni", cluster)) for cluster in CLUSTERS}
        for cluster, args in runs.items():
            config = launch.load_yaml(ROOT / "experiments" / "cluster" / f"{cluster}.yaml")
            for key in PATH_KEYS - {"labels_path"}:
                self.assertEqual(args[key], config[key], f"{cluster}: {key}")
            # The labels with the demographics columns, next to each cluster's labels file.
            demographics = Path(config["labels_path"]).with_name("labels_cleaned_3class_demog.csv")
            self.assertEqual(args["labels_path"], str(demographics), cluster)
        changed = {key for key in runs["runai"] if runs["runai"][key] != runs["slurm"][key]}
        self.assertEqual(changed, PATH_KEYS | {"out_dir"})
        for cluster in CLUSTERS:  # inherited through the base's _cluster block
            gumbel = parsed(resolve("adni_gumbel", cluster))
            self.assertEqual(
                {key for key in gumbel if gumbel[key] != runs[cluster][key]}, {"mask_mode", "mask_lr_scale", "model_id"}
            )

    def test_adni_default_model_is_the_synthetic_reference(self):
        args = parsed(resolve("adni"))
        self.assertEqual(
            (args["encoder_architecture"], args["conv_readout"], args["norm_type"]), ("conv", "mlp", "group")
        )
        self.assertEqual((args["latent_dim"], args["content_channels"], args["encoder_head_hidden"]), (12, 9, 100))
        self.assertEqual((args["best_metric"], args["asymmetric_aug"], args["num_workers"]), ("val_loss", True, 8))
        reference = json.loads((ROOT / "experiments" / "encoder_comparison.json").read_text())["shared"]
        for key in ("hidden_channels", "res_channels", "nb_res_layers", "downscale_factor", "batch_size", "tau", "lr"):
            self.assertEqual(args[key], reference[key], key)


class CommandTests(unittest.TestCase):
    def test_every_trainer_option_round_trips(self):
        """Any option reaches the trainer with --set, so no flag needs launcher code of its own."""
        for action in encoder_args.build_parser()._actions:
            if not action.option_strings or action.dest == "help":
                continue
            if isinstance(action, argparse.BooleanOptionalAction) or action.nargs == 0:
                value = not action.default
            elif action.choices:
                value = next(choice for choice in action.choices if choice != action.default)
            elif action.nargs in (3, "+"):
                value = [1, 2, 3]
            elif action.type in (int, float):
                value = action.type((action.default or 0) + 1)
            else:
                value = "x"
            with self.subTest(option=action.dest):
                argv = encoder_args.cli_arguments({action.dest: value})
                self.assertEqual(getattr(encoder_args.build_parser().parse_args(argv), action.dest), value)

    def test_launch_refuses_what_the_trainer_refuses(self):
        message = exits(
            self,
            2,
            launch.encoder_cli_args,
            resolve("conv_mlp", "local", mask_mode="learned", contrastive_loss_type="barlow_twins"),
        )
        self.assertIn("conv_mlp_s42: the trainer refuses this configuration: --mask-mode learned needs", message)
        self.assertNotIn("usage:", message)
        self.assertIn("Real data needs --dataroot", exits(self, 2, launch.encoder_cli_args, resolve("adni", "local")))
        self.assertIn(
            "no option for 'learning_rate'", exits(self, 1, launch.encoder_cli_args, resolve("conv", learning_rate=1))
        )
        self.assertIn("with tag", exits(self, 1, launch.encoder_cli_args, resolve("conv", model_id="x")))
        self.assertIn("give true or false", exits(self, 1, launch.encoder_cli_args, resolve("conv", cache="no")))

    def test_runai_command_and_job_name(self):
        command = launch.build_encoder_runai_command(resolve("adni", "runai", mask_mode="learned"))
        self.assertEqual(command[:5], ["runai", "training", "standard", "submit", "encoder-adni-conv-mlp-s42"])
        runai = launch.load_yaml(ROOT / "experiments" / "cluster" / "runai.yaml")["_runai"]
        expected = [part for pair in launch._runai_submit_flags(runai) for part in pair if part is not None]
        self.assertEqual(command[5 : 5 + len(expected)], expected)
        body = command[-1]
        self.assertTrue(body.startswith(f"cd {runai['repo_path']} ||"))
        self.assertIn("torch.cuda.is_available()", body)
        args = encoder_args.parse_args(trainer_command(shlex.split(body)))
        self.assertEqual((args.mask_mode, args.model_id), ("learned", "encoder_adni_conv_mlp_s42"))
        self.assertEqual(launch.runai_job_name("Conv_MLP_s42"), "conv-mlp-s42")
        exits(self, 1, launch.runai_job_name, "bad.name")

    def test_local_launch_uses_this_python_and_the_lpips_stand_in(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch.tempfile, "gettempdir", return_value=tmp):
            with patch.object(launch.importlib.util, "find_spec", return_value=None):
                command, env = launch.encoder_local_launch(resolve("conv_mlp", "local", device="mps"))
            if command[0] == "caffeinate":
                command = command[2:]
            self.assertEqual(command[:3], [sys.executable, "-m", launch.ENCODER_ENTRYPOINT])
            self.assertEqual(encoder_args.parse_args(command[3:]).device, "mps")
            self.assertEqual((env["PYTORCH_ENABLE_MPS_FALLBACK"], env["OMP_NUM_THREADS"]), ("1", "1"))
            stand_in, repo = env["PYTHONPATH"].split(launch.os.pathsep)[:2]
            self.assertEqual((Path(stand_in), Path(repo)), (Path(tmp) / "multiview-crl-lpips-stand-in", ROOT))
            check = "import lpips\ntry:\n    lpips.LPIPS()\nexcept ImportError:\n    print('refused')"
            result = subprocess.run(
                [sys.executable, "-c", check], env={"PYTHONPATH": stand_in}, capture_output=True, text=True
            )
            self.assertEqual(result.stdout.strip(), "refused")


class ScriptTests(unittest.TestCase):
    def test_committed_runai_scripts_are_current_and_preview_a_valid_command(self):
        """Each experiment's script is exactly what launch.py writes, so nothing was edited by hand."""
        committed = sorted((ROOT / "experiments" / "generated" / "encoder").glob("*.runai.sh"))
        self.assertEqual(
            [p.name.split(".")[0] for p in committed], sorted(p.stem for p in launch.find_experiment_yamls(ENCODER))
        )
        for script in committed:
            name = script.name.split(".")[0]
            with self.subTest(script=script.name):
                path = ENCODER / f"{name}.yaml"
                regenerate = f"python scripts/launch.py experiments/encoder/{name}.yaml --cluster runai --generate"
                expected = launch.build_encoder_script(
                    launch.resolve_config(path, "runai", {}), "runai", path, regenerate
                )
                self.assertEqual(script.read_text(), expected)
                preview = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True, text=True, check=True)
                args = encoder_args.parse_args(trainer_command(shlex.split(preview.stdout)))
                self.assertEqual(args.model_id, launch.resolve_config(path, "runai", {})["tag"])

    def test_slurm_script_previews_never_installs_and_refuses_to_run_outside_a_job(self):
        for cluster in ("slurm", "slurm_bio"):
            with self.subTest(cluster=cluster), tempfile.TemporaryDirectory() as tmp:
                config = resolve("adni", cluster)
                script = Path(tmp) / "run.sh"
                script.write_text(launch.build_encoder_script(config, cluster, ENCODER / "adni.yaml", "regenerate"))
                text = script.read_text()
                constraint = launch.load_yaml(ROOT / "experiments" / "cluster" / f"{cluster}.yaml")["_slurm"][
                    "constraint"
                ]
                self.assertIn(f"#SBATCH --constraint={constraint}\n", text)
                self.assertNotIn("pip install", text)
                self.assertNotIn("conda create", text)
                preview = subprocess.run(["bash", str(script), "--dry-run"], capture_output=True, text=True, check=True)
                args = encoder_args.parse_args(trainer_command(shlex.split(preview.stdout)))
                self.assertEqual(args.out_dir, "/scratch/users/k24058220/encoder_adni/runs")
                outside = {"PATH": "/usr/bin:/bin", "HOME": tmp}  # no SLURM_JOB_ID
                refused = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=outside)
                self.assertEqual(refused.returncode, 2)
                self.assertIn("Submit with sbatch", refused.stderr)

    def test_generate_writes_one_overridden_experiment_or_a_whole_directory(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch, "GENERATED_DIR", Path(tmp)):
            argv = ["launch.py", str(ENCODER / "adni.yaml"), "--cluster", "runai", "--generate"]
            argv += ["--set", "mask_mode=learned", "tag=encoder_adni_gumbel"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                launch.main()
            script = Path(tmp) / "encoder" / "encoder_adni_gumbel.runai.sh"
            text = script.read_text()
            self.assertIn("runai training standard submit encoder-adni-gumbel \\", text)
            self.assertIn("--mask-mode learned", text)
            self.assertIn("--set mask_mode=learned tag=encoder_adni_gumbel", text.splitlines()[2])
            with patch.object(sys, "argv", ["launch.py", str(ENCODER), "--cluster", "slurm", "--generate"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    launch.main()
            written = {path.name.split(".")[0] for path in (Path(tmp) / "encoder").glob("*.slurm.sh")}
            self.assertEqual(written, {path.stem for path in launch.find_experiment_yamls(ENCODER)})
            with patch.object(sys, "argv", ["launch.py", str(ENCODER), "--generate", "--set", "seed=1"]):
                exits(self, 2, launch.main)

    def test_dry_run_prints_and_never_submits(self):
        for cluster, marker in (("runai", "RUNAI COMMAND:"), ("slurm", "SLURM SCRIPT:"), ("local", "COMMAND:")):
            with self.subTest(cluster=cluster), patch.object(launch.os, "execvpe") as execute:
                argv = ["launch.py", str(ENCODER / "conv_mlp.yaml"), "--cluster", cluster, "--dry-run"]
                with patch.object(launch.subprocess, "run", wraps=launch.subprocess.run) as run:
                    with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as output:
                        launch.main()
                self.assertIn(marker, output.getvalue())
                self.assertIn("--model-id conv_mlp_s42", output.getvalue())
                # Only the provenance snapshot's git calls run; nothing is submitted or started.
                self.assertTrue(all(call.args[0][0] == "git" for call in run.call_args_list))
                execute.assert_not_called()

    def test_from_config_relaunches_an_encoder_run(self):
        settings = parsed(resolve("conv_mlp_patch"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({**settings, "retired_flag": 1}))
            with contextlib.redirect_stderr(io.StringIO()) as warnings:
                config = launch.load_from_config(path)
        self.assertIn("retired_flag", warnings.getvalue())
        self.assertTrue(launch.is_encoder(config))
        self.assertEqual(config["tag"], "conv_mlp_patch_s42")
        self.assertEqual(parsed(config), settings)


if __name__ == "__main__":
    unittest.main()
