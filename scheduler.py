# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


class Timeline:
    # An interval goes from inclusive to exclusive: [n,n+1).
    # An interval is identified by its left border: interval n = [n,n+1)
    # For timesteps (0.0, 1.0, 1.5, 2.0), we have three intervals:
    # 0: [0.0, 1.0)
    # 1: [1.0, 2.0)
    # 2: [2.0, 3.0)

    def __init__(self, settings, renderer, data_handler):
        self.tracking_mode = settings.tracking_mode
        self.optimization_mode = settings.optimization_mode
        self.timeline_range = data_handler.get_timeline_range()

        self.timestep_batch = data_handler.get_batch(
            subset_name="top_left_corner"
        )  # one ray per image. among all training images.
        self.pre_correction = renderer.pre_correction

        self.do_nrnerf = settings.do_nrnerf
        self.do_dnerf = settings.do_dnerf

        self.num_intervals = int(np.ceil(self.timeline_range))
        if np.ceil(self.timeline_range) == np.floor(self.timeline_range):
            self.num_intervals += 1

        self.state = {}

        self.state["newly_active_intervals"] = None
        self.state["already_active_intervals"] = None

        self.state["corrected_intervals_to_images"] = None
        self.recompute_corrected_assingment_every = 20000

        self.initial_iterations = 20000
        self.extend_every_n_iterations = 10000

        if self.tracking_mode == "temporal":
            # in unnormalized timesteps
            if self.optimization_mode == "per_timestep":
                self.initial_active_intervals = 1
                self.extend_by = 1
            elif self.optimization_mode == "one_rest":
                self.initial_active_intervals = 1
                self.extend_by = 10000
                self.initial_iterations = 20000
                self.extend_every_n_iterations = (self.get_num_intervals() - 1) * 10000
            elif self.optimization_mode == "dnerf":
                self.initial_active_intervals = 3
                self.extend_by = 1
                per_interval = (
                    (1.0 / 8.0) * self.get_num_training_iterations(settings.num_iterations)
                ) // self.get_num_intervals()
                self.initial_iterations = 3 * per_interval
                self.extend_every_n_iterations = per_interval

            self.state["previous_active_intervals"] = 0  # init
            self.state["all_have_been_active_since_iteration"] = None  # init

        elif self.tracking_mode == "plain":
            pass

    # INTERVALS

    def get_num_intervals(self):
        return self.num_intervals

    def get_newly_active_intervals(self):
        # sorted from early to late intervals
        return self.state["newly_active_intervals"]

    def get_already_active_intervals(self):
        # sorted from early to late intervals
        return self.state["already_active_intervals"]

    def active_intervals_changed(self):
        return self._active_intervals_changed

    def update(self, training_iteration, log=None):

        self._active_intervals_changed = False
        self._intervals_to_images_changed = False

        if self.tracking_mode == "temporal":

            if self.state["all_have_been_active_since_iteration"] is None or (
                self.state["all_have_been_active_since_iteration"] is not None
                and training_iteration - self.state["all_have_been_active_since_iteration"]
                <= self.extend_every_n_iterations
            ):

                current_active_intervals = (
                    self.initial_active_intervals
                    + (
                        (
                            (
                                max(
                                    0,
                                    training_iteration
                                    - self.initial_iterations
                                    + self.extend_every_n_iterations,
                                )
                            )
                            // self.extend_every_n_iterations
                        )
                    )
                    * self.extend_by
                )

                if current_active_intervals >= self.num_intervals:
                    current_active_intervals = self.num_intervals
                    if self.state["all_have_been_active_since_iteration"] is None:
                        self.state["all_have_been_active_since_iteration"] = training_iteration

                if self.state["previous_active_intervals"] != current_active_intervals or (
                    self.state["all_have_been_active_since_iteration"] is not None
                    and training_iteration - self.state["all_have_been_active_since_iteration"]
                    == self.extend_every_n_iterations
                ):

                    self.state["already_active_intervals"] = torch.arange(
                        self.state["previous_active_intervals"], dtype=np.long
                    )
                    self.state["newly_active_intervals"] = torch.arange(
                        start=self.state["previous_active_intervals"],
                        end=current_active_intervals,
                        dtype=np.long,
                    )
                    self._active_intervals_changed = True

                    self.state["previous_active_intervals"] = current_active_intervals

        elif self.tracking_mode == "plain":

            if self.state["newly_active_intervals"] is None:
                self.state["newly_active_intervals"] = torch.arange(0, dtype=np.long)
                self.state["already_active_intervals"] = torch.arange(
                    self.num_intervals, dtype=np.long
                )
                self._active_intervals_changed = True

        else:
            raise NotImplementedError

        if (
            self._active_intervals_changed
            or training_iteration % self.recompute_corrected_assingment_every == 0
        ):
            self.compute_corrected_intervals_to_images()

    # IMAGES

    def intervals_to_images_changed(self):
        return self._intervals_to_images_changed

    def get_intervals_to_images_corrected(self):
        return self.state["corrected_intervals_to_images"]

    def get_images_in_intervals(self, intervals):
        device = intervals.device
        if len(intervals) == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        intervals_to_images = self.get_intervals_to_images_corrected()
        interval_images = [intervals_to_images[interval].long() for interval in intervals]
        if all(len(interval) == 0 for interval in interval_images):
            return torch.empty((0,), dtype=torch.long, device=device)
        return torch.cat(interval_images, dim=0).to(device)

    def compute_corrected_intervals_to_images(self):
        timestep_batch = {key: tensor.clone() for key, tensor in self.timestep_batch.items()}
        with torch.no_grad():
            images_to_corrected_timesteps = self.pre_correction(timestep_batch)["timesteps"]
            images_to_corrected_timesteps *= (
                self.timeline_range
            )  # from normalized to unnormalized timesteps

        hotfix = True  # numerics lead to incorrect integer part/rounding
        if hotfix:
            preliminary_intervals = torch.floor(images_to_corrected_timesteps).long()
            mask = (images_to_corrected_timesteps - preliminary_intervals) > 0.9999
            images_to_corrected_timesteps[mask] += 0.0001

        images_to_corrected_intervals = torch.floor(images_to_corrected_timesteps).long()
        device = images_to_corrected_timesteps.device

        corrected_intervals_to_images = [[] for _ in range(self.num_intervals)]
        for imageid, interval in enumerate(images_to_corrected_intervals):  # expensive
            corrected_intervals_to_images[interval].append(imageid)
        corrected_intervals_to_images = [
            torch.from_numpy(np.array(imageids)).to(device)
            for imageids in corrected_intervals_to_images
        ]

        self.state["corrected_intervals_to_images"] = corrected_intervals_to_images
        self._intervals_to_images_changed = True

    # TRAINING ITERATIONS

    def get_num_training_iterations(self, settings_num_iterations):
        if settings_num_iterations > 0:
            return settings_num_iterations
        elif self.do_nrnerf:
            return max(
                200000,
                int(
                    float(
                        self.initial_iterations
                        + (self.get_num_intervals() - 1) * self.extend_every_n_iterations
                    )
                    / 3.0
                ),
            )
        elif self.optimization_mode == "one_rest":
            return self.initial_iterations + self.extend_every_n_iterations
        else:
            return (
                self.initial_iterations
                + (self.get_num_intervals() - 1) * self.extend_every_n_iterations
            )  # first interval is taken care off by initial_iterations

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state = state_dict


class BatchComposition:
    def __init__(self, settings, timeline, data_handler):

        self.timeline = timeline
        self.batch_composer = data_handler.batch_composer
        self.optimization_mode = settings.optimization_mode

        self.mode = "plain"  # "alternating" (don't update canonical model for newly_active batches), "plain"

        self._batch_composition = None  # init

    def schedule(self, training_iteration):

        if (
            training_iteration == 0
            or self.timeline.active_intervals_changed()
            or self.timeline.intervals_to_images_changed()
        ):

            newly_active_intervals = self.timeline.get_newly_active_intervals()
            already_active_intervals = self.timeline.get_already_active_intervals()

            newly_active_images = self.timeline.get_images_in_intervals(newly_active_intervals)
            already_active_images = self.timeline.get_images_in_intervals(
                already_active_intervals
            )  # among all training images

            if self.optimization_mode == "dnerf":
                self._batch_composition = [
                    {
                        "name": "all_active",
                        "fraction": 1.0,
                        "imageids": torch.cat([newly_active_images, already_active_images], dim=0),
                    }
                ]
            else:
                self._batch_composition = [
                    {
                        "name": "newly_active",
                        "fraction": 1.0
                        if self.optimization_mode in ["per_timestep", "one_rest"]
                        else 0.5,
                        "imageids": newly_active_images,
                    },
                    {
                        "name": "already_active",
                        "fraction": 0.0
                        if self.optimization_mode in ["per_timestep", "one_rest"]
                        else 0.5,
                        "imageids": already_active_images,
                    },
                ]

            LOGGER.info(self._batch_composition)

        if self.mode == "plain":
            self.batch_composition = self._batch_composition
        elif self.mode == "alternating":

            if self.optimization_mode == "per_timestep":
                raise NotImplementedError  # not compatible with how updates to canonical model are handled
            else:
                if len(self._batch_composition[0]["imageids"]) > 0 and len(
                    self._batch_composition[1]["imageids"]
                ):
                    self._batch_composition_index = training_iteration % 2
                    self.batch_composition = [
                        self._batch_composition[self._batch_composition_index]
                    ]
                else:
                    self._batch_composition_index = None
                    self.batch_composition = self._batch_composition

        self.batch_composer.set_batch_composition(
            self.batch_composition
        )  # takes care of continuing from checkpoints

    def state_dict(self):
        return {"_batch_composition": self._batch_composition}

    def load_state_dict(self, state_dict):
        self._batch_composition = state_dict["_batch_composition"]


class OnlyLoadActiveTrainingImages:
    def __init__(self, settings, timeline, data_handler):

        self.timeline = timeline
        self.data_handler = data_handler

        self.do_only_load_active_training_images = not settings.always_load_full_dataset
        self.factor = settings.factor
        self.optimization_mode = settings.optimization_mode

        self.num_total_rays_to_precompute = int(1e9)

        self.active_imageids = None

    def schedule(self):

        if not self.do_only_load_active_training_images:
            return

        if (
            self.active_imageids is None
            or self.timeline.active_intervals_changed()
            or self.timeline.intervals_to_images_changed()
        ):

            newly_active_intervals = self.timeline.get_newly_active_intervals()
            already_active_intervals = self.timeline.get_already_active_intervals()

            newly_active_images = self.timeline.get_images_in_intervals(newly_active_intervals)
            already_active_images = self.timeline.get_images_in_intervals(already_active_intervals)

            if self.optimization_mode in ["per_timestep", "one_rest"]:
                relevant_images = newly_active_images
            else:
                relevant_images = torch.cat(
                    [newly_active_images, already_active_images], dim=0
                )  # imageids among training images
            all_train_imageids = self.data_handler.get_train_imageids()

            active_imageids = all_train_imageids[relevant_images]  # among all images

            if (
                self.active_imageids is not None
                and self.active_imageids.shape == active_imageids.shape
                and np.all(self.active_imageids == active_imageids)
            ):
                return

            self.active_imageids = active_imageids

            self.data_handler.load_training_set(
                factor=self.factor,
                num_total_rays_to_precompute=self.num_total_rays_to_precompute,
                foreground_focused=True,
                imageids=active_imageids,
            )

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class LearningRateScheduler:
    def __init__(self, settings, optimizer, timeline):

        self.optimizer = optimizer
        self.timeline = timeline

        self.warmup_iterations = 1000
        self.warmup_factor = 0.01

        self.fix_mlps_after_iterations = 0

        self.deformation_per_timestep_decay_rate = settings.deformation_per_timestep_decay_rate
        self.slow_canonical_per_timestep_learning_rate = (
            settings.slow_canonical_per_timestep_learning_rate
        )

        self.fix_coarse_after_a_while = settings.fix_coarse_after_a_while
        self.let_canonical_vary_at_last = settings.let_canonical_vary_at_last
        self.let_only_brightness_vary = settings.let_only_brightness_vary
        self.fix_everything_except_for_brightness_at_the_end = False

        self.variant = settings.variant

        self.keep_coarse_mlp_constant = settings.keep_coarse_mlp_constant
        self.optimization_mode = settings.optimization_mode

        self.do_pref = settings.do_pref
        self.do_nrnerf = settings.do_nrnerf
        self.do_dnerf = settings.do_dnerf

        self.use_global_transform = settings.use_global_transform

    def schedule(self, training_iteration, batch_composition=None):

        for optimizer_with_info in self.optimizer.optimizers_with_information:

            initial_learning_rate = optimizer_with_info["initial_learning_rate"]
            decay_rate = optimizer_with_info["decay_rate"]
            decay_steps = optimizer_with_info["decay_steps"]
            tags = optimizer_with_info["tags"]
            name = optimizer_with_info["name"]

            if self.do_pref:
                initial_learning_rate = 5e-4
                decay_rate = 0.01
                decay_steps = 50000
            if self.do_nrnerf or self.do_dnerf:
                initial_learning_rate = 5e-4
                decay_rate = 0.1
                decay_steps = self.timeline.get_num_training_iterations(0)
            if self.optimization_mode == "one_rest":
                decay_steps = self.timeline.get_num_training_iterations(0)

            if self.optimization_mode in ["per_timestep", "one_rest"]:
                initial_iterations = self.timeline.initial_iterations
                extend_every_n_iterations = self.timeline.extend_every_n_iterations

                let_canonical_vary = False
                keep_latents_slow = False
                variants_use_fixed_learning_rate_for_canonical = False
                reset_optimizer_for_each_timestep = False  # True gives very bad results

                this_iteration = (
                    training_iteration - initial_iterations
                ) % extend_every_n_iterations

                if training_iteration < initial_iterations:
                    decay_rate = 0.01
                    decay_steps = initial_iterations
                else:
                    if "deformation" in tags:
                        if self.keep_coarse_mlp_constant:
                            if name == "pure_mlp_bending_model":
                                decay_rate = 1.0
                        decay_rate = self.deformation_per_timestep_decay_rate
                    elif self.variant in ["snfa", "snfag"] and "canonical" in tags:
                        if variants_use_fixed_learning_rate_for_canonical:
                            decay_rate = 1.0
                        else:
                            decay_rate = 0.1
                    elif let_canonical_vary and "canonical" in tags:
                        decay_rate = 1.0
                    elif self.let_canonical_vary_at_last and "canonical" in tags:
                        decay_rate = 1.0
                    elif keep_latents_slow and name == "latent_codes":
                        decay_rate = 0.1

                if reset_optimizer_for_each_timestep:
                    if this_iteration == 0:
                        optimizer = optimizer_with_info["optimizer"]
                        for group in optimizer.param_groups:
                            for p in group["params"]:
                                optimizer.state[p] = {}

            # exponential long-term learning rate decay
            if decay_steps > 0:
                new_lrate = initial_learning_rate * (
                    decay_rate ** (min(1.0, training_iteration / decay_steps))
                )

                if self.optimization_mode == "per_timestep":
                    if training_iteration > initial_iterations:
                        if "deformation" in tags:
                            decay_steps = extend_every_n_iterations
                            if self.fix_coarse_after_a_while and "coarse_deformation" in tags:
                                if self.let_canonical_vary_at_last or self.variant in [
                                    "snfa",
                                    "snfag",
                                ]:
                                    fraction = 1.0 / 3.0
                                else:
                                    fraction = 1.0 / 2.0
                                decay_steps = fraction * extend_every_n_iterations
                            new_lrate = initial_learning_rate * (
                                decay_rate ** (min(1.0, this_iteration / decay_steps))
                            )
                        elif self.variant in ["snfa", "snfag"] and "canonical" in tags:
                            if variants_use_fixed_learning_rate_for_canonical:
                                new_lrate = self.slow_canonical_per_timestep_learning_rate
                            else:
                                initial_learning_rate = 1e-2
                                offset = (2.0 / 3.0) * extend_every_n_iterations
                                decay_steps = (1.0 / 3.0) * extend_every_n_iterations
                                new_lrate = initial_learning_rate * (
                                    decay_rate
                                    ** (min(1.0, (this_iteration - offset) / decay_steps))
                                )
                        elif let_canonical_vary and "canonical" in tags:
                            new_lrate = self.slow_canonical_per_timestep_learning_rate
                        elif self.let_canonical_vary_at_last and "canonical" in tags:
                            new_lrate = self.slow_canonical_per_timestep_learning_rate
                        elif keep_latents_slow and name == "latent_codes":
                            decay_steps = extend_every_n_iterations
                            initial_learning_rate = 1e-5
                            new_lrate = initial_learning_rate * (
                                decay_rate ** (min(1.0, this_iteration / decay_steps))
                            )

            else:
                new_lrate = initial_learning_rate

            # exponential short-term warming up
            if training_iteration < self.warmup_iterations:
                # in case images are very dark or very bright, need to keep network from initially building up too much momentum
                new_lrate *= self.warmup_factor ** (
                    1.0 - (training_iteration / self.warmup_iterations)
                )

            def freeze_parameters():
                for param in optimizer_with_info["parameters"]:
                    param.grad = None
                return 0.0

            # fix MLPs after some iterations
            if "mlp" in tags and self.fix_mlps_after_iterations > 0:
                if training_iteration > self.fix_mlps_after_iterations:
                    new_lrate = freeze_parameters()

            if self.optimization_mode == "per_timestep":
                if "canonical" in tags:
                    if training_iteration <= initial_iterations:
                        if "separate_brightness" in tags:
                            network_could_die = self.timeline.initial_active_intervals == 1
                            if network_could_die:
                                new_lrate = freeze_parameters()
                    else:
                        if self.variant == "snfa" and "separate_appearance" in tags:
                            if this_iteration < (2.0 / 3.0) * extend_every_n_iterations:
                                new_lrate = freeze_parameters()
                        elif self.variant == "snfag":
                            if this_iteration < (2.0 / 3.0) * extend_every_n_iterations:
                                new_lrate = freeze_parameters()
                        elif let_canonical_vary:
                            pass
                        elif self.let_canonical_vary_at_last:

                            if self.let_only_brightness_vary and "separate_brightness" not in tags:
                                new_lrate = freeze_parameters()
                            else:
                                if this_iteration < (2.0 / 3.0) * extend_every_n_iterations:
                                    new_lrate = 0.0  # still want to already accumulate momentum

                        else:
                            new_lrate = freeze_parameters()
                if "deformation" in tags:
                    if training_iteration <= initial_iterations:
                        deformation_network_could_die = self.timeline.initial_active_intervals == 1
                        if deformation_network_could_die:
                            new_lrate = freeze_parameters()
                    else:
                        if self.fix_coarse_after_a_while:
                            iterations_for_global_transform = 1000
                            if "coarse_deformation" in tags:
                                if self.let_canonical_vary_at_last or self.variant in [
                                    "snfa",
                                    "snfag",
                                ]:
                                    fraction = 1.0 / 3.0
                                else:
                                    fraction = 1.0 / 2.0
                                if this_iteration > fraction * extend_every_n_iterations:
                                    new_lrate = freeze_parameters()
                                if (
                                    self.use_global_transform
                                    and this_iteration <= iterations_for_global_transform
                                ):
                                    new_lrate = freeze_parameters()
                            if "global_transform" in tags:
                                assert self.use_global_transform
                                if this_iteration > iterations_for_global_transform:
                                    new_lrate = freeze_parameters()
                            if "fine_deformation" in tags:
                                if self.variant in ["snfa", "snfag"]:
                                    if this_iteration < (1.0 / 3.0) * extend_every_n_iterations:
                                        new_lrate = freeze_parameters()
                                    if this_iteration > (2.0 / 3.0) * extend_every_n_iterations:
                                        new_lrate = freeze_parameters()
                if "vignetting" in tags:
                    if training_iteration > initial_iterations:
                        new_lrate = freeze_parameters()

                if (
                    self.fix_everything_except_for_brightness_at_the_end
                    and self.fix_coarse_after_a_while
                    and self.let_canonical_vary_at_last
                ):
                    if "separate_brightness" not in tags:
                        if training_iteration > initial_iterations:
                            fraction = 2.0 / 3.0
                            if this_iteration > fraction * extend_every_n_iterations:
                                new_lrate = freeze_parameters()

            if batch_composition is not None:
                if batch_composition.mode == "alternating" and "canonical" in tags:
                    if (
                        batch_composition._batch_composition_index is not None
                    ):  # already seen images are also active
                        if batch_composition._batch_composition_index == 0:  # newly_active batch
                            new_lrate = freeze_parameters()

            for param_group in optimizer_with_info["optimizer"].param_groups:
                param_group["lr"] = new_lrate

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class ZeroOutFineDeformations:
    def __init__(self, settings, timeline, scene):

        self.do_zero_out = settings.do_zero_out
        self.optimization_mode = settings.optimization_mode
        self.do_variant = settings.variant in ["snfa", "snfag"]

        self.timeline = timeline
        self.scene = scene

    def schedule(self, training_iteration, learning_rate_scheduler):

        if not self.do_zero_out:
            return

        if self.optimization_mode == "per_timestep":
            initial_iterations = self.timeline.initial_iterations
            extend_every_n_iterations = self.timeline.extend_every_n_iterations

            this_iteration = (training_iteration - initial_iterations) % extend_every_n_iterations

            fix_coarse_after_a_while = learning_rate_scheduler.fix_coarse_after_a_while
            let_canonical_vary_at_last = learning_rate_scheduler.let_canonical_vary_at_last

            zero_out = False
            if fix_coarse_after_a_while:
                if let_canonical_vary_at_last or self.do_variant:
                    fraction = 1.0 / 3.0
                else:
                    fraction = 1.0 / 2.0
                if this_iteration < fraction * extend_every_n_iterations:
                    zero_out = True

            self.scene.deformation_model.zero_out_fine_deformations = zero_out

    def reset_for_rendering(self):
        if self.do_zero_out:
            self.scene.deformation_model.zero_out_fine_deformations = False

    def fine_deformations(self, zero_out):
        self.scene.deformation_model.zero_out_fine_deformations = zero_out

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class CoarseSmoothnessLoss:
    def __init__(self, settings, timeline, trainer, scene):

        self.do_set_weight = (
            scene.deformation_model is not None
            and scene.deformation_model.coarse_and_fine
            and (
                settings.weight_coarse_smooth_deformations > 0.0
                or settings.weight_fine_smooth_deformations > 0.0
            )
        )

        self.timeline = timeline
        self.losses = trainer.losses

        self.optimization_mode = settings.optimization_mode
        self.do_zero_out = settings.do_zero_out  # and not (
        self.do_variant = settings.variant in ["snfa", "snfag"]

        self.original_coarse_weight = settings.weight_coarse_smooth_deformations
        self.original_fine_weight = settings.weight_fine_smooth_deformations

    def schedule(self, training_iteration, learning_rate_scheduler):

        if not self.do_set_weight:
            return

        if self.optimization_mode == "per_timestep":
            initial_iterations = self.timeline.initial_iterations
            extend_every_n_iterations = self.timeline.extend_every_n_iterations

            this_iteration = (training_iteration - initial_iterations) % extend_every_n_iterations

            fix_coarse_after_a_while = learning_rate_scheduler.fix_coarse_after_a_while
            let_canonical_vary_at_last = learning_rate_scheduler.let_canonical_vary_at_last

            coarse_weight = self.original_coarse_weight
            fine_weight = self.original_fine_weight

            if fix_coarse_after_a_while:
                if let_canonical_vary_at_last or self.do_variant:
                    fraction = 1.0 / 3.0
                else:
                    fraction = 1.0 / 2.0
                if this_iteration < fraction * extend_every_n_iterations:
                    if self.do_zero_out:
                        fine_weight = 0.0
                if this_iteration > fraction * extend_every_n_iterations:
                    coarse_weight = 0.0
                if this_iteration > (2.0 / 3.0) * extend_every_n_iterations:
                    if self.do_variant:
                        fine_weight = 0.0

            if training_iteration < initial_iterations:
                coarse_weight = 0.0
                fine_weight = 0.0

            self.losses.weight_coarse_smooth_deformations = coarse_weight
            self.losses.weight_fine_smooth_deformations = fine_weight

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class LoadCurrentPruningVoxelGrid:
    def __init__(self, settings, timeline, scene, data_handler):

        self.use_pruning = settings.use_pruning
        self.optimization_mode = settings.optimization_mode
        self.tracking_mode = settings.tracking_mode

        self.timeline = timeline
        self.pruning = scene.pruning
        self.data_handler = data_handler

        self.initial_probabilistic_pruning = 0.0  # default: 0.0
        self.original_no_pruning_probability = settings.no_pruning_probability

    def schedule(self, training_iteration):

        if not self.use_pruning:
            return

        if self.optimization_mode == "per_timestep" and self.tracking_mode == "temporal":

            initial_iterations = self.timeline.initial_iterations
            extend_every_n_iterations = self.timeline.extend_every_n_iterations

            this_iteration = (training_iteration - initial_iterations) % extend_every_n_iterations

            if training_iteration < initial_iterations:
                if self.initial_probabilistic_pruning > 0.0:
                    if self.pruning.voxel_grid is None:
                        self.pruning.no_pruning_probability = (
                            1.0 - self.initial_probabilistic_pruning
                        )
                        timestep = float(self.data_handler.precomputed["timesteps"].numpy()[0])
                        self.pruning.load_voxel_grid(timestep)
                else:
                    self.pruning.voxel_grid = None
            elif self.pruning.voxel_grid is None or this_iteration == 0:
                self.pruning.no_pruning_probability = self.original_no_pruning_probability
                timestep = float(self.data_handler.precomputed["timesteps"].numpy()[0])
                self.pruning.load_voxel_grid(timestep)

        else:
            if self.pruning.voxel_grid is None:
                self.pruning.load_voxel_grid("all")

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class CanonicalModelRegularizationLosses:
    def __init__(self, settings, timeline, trainer):

        self.timeline = timeline
        self.losses = trainer.losses

        self.optimization_mode = settings.optimization_mode
        self.original_weight_hard_surface_loss = settings.weight_hard_surface_loss
        self.original_weight_background_loss = settings.weight_background_loss

        self.variant = settings.variant

        self.schedule_type = "constant"

    def schedule(self, training_iteration, learning_rate_scheduler):

        if (
            self.original_weight_hard_surface_loss == 0.0
            and self.original_weight_background_loss == 0.0
        ):
            return

        if self.optimization_mode in ["per_timestep", "one_rest"]:
            initial_iterations = self.timeline.initial_iterations
            extend_every_n_iterations = self.timeline.extend_every_n_iterations

            this_iteration = (training_iteration - initial_iterations) % extend_every_n_iterations

            fix_coarse_after_a_while = learning_rate_scheduler.fix_coarse_after_a_while
            let_canonical_vary_at_last = learning_rate_scheduler.let_canonical_vary_at_last

            if training_iteration <= initial_iterations:
                keep_original = True
            else:
                keep_original = False
                if self.variant == "snfag":
                    keep_original = True
                if fix_coarse_after_a_while:
                    if let_canonical_vary_at_last:
                        fraction = 2.0 / 3.0
                        if this_iteration > fraction * extend_every_n_iterations:
                            keep_original = True

            if keep_original:
                change_factor = 0.1
                if self.schedule_type == "constant":
                    factor = 1.0

                elif self.schedule_type == "decrease":
                    convex = training_iteration / initial_iterations
                    factor = 1.0 * (1.0 - convex) + change_factor * convex

                elif self.schedule_type == "u_shape":
                    if training_iteration < 0.5 * initial_iterations:
                        convex = training_iteration / (0.5 * initial_iterations)
                        factor = 1.0 * (1.0 - convex) + change_factor * convex
                    else:
                        convex = (training_iteration - 0.5 * initial_iterations) / (
                            0.5 * initial_iterations
                        )
                        factor = 1.0 * convex + change_factor * (1.0 - convex)

                elif self.schedule_type == "hill":
                    if training_iteration < 0.5 * initial_iterations:
                        convex = training_iteration / (0.5 * initial_iterations)
                        factor = 1.0 * convex + change_factor * (1.0 - convex)
                    else:
                        convex = (training_iteration - 0.5 * initial_iterations) / (
                            0.5 * initial_iterations
                        )
                        factor = 1.0 * (1.0 - convex) + change_factor * convex

                self.losses.weight_hard_surface_loss = (
                    factor * self.original_weight_hard_surface_loss
                )
                self.losses.weight_background_loss = factor * self.original_weight_background_loss
            else:
                self.losses.weight_hard_surface_loss = 0.0
                self.losses.weight_background_loss = 0.0

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class VariantsReconstructionLosses:
    def __init__(self, settings, timeline, trainer):

        self.timeline = timeline
        self.losses = trainer.losses

        self.optimization_mode = settings.optimization_mode

        self.variant = settings.variant

    def schedule(self, training_iteration):

        self.losses.variant_reconstruction_mode = None

        if self.variant in ["snfa", "snfag"] and self.optimization_mode in [
            "per_timestep",
            "one_rest",
        ]:
            initial_iterations = self.timeline.initial_iterations
            extend_every_n_iterations = self.timeline.extend_every_n_iterations

            if training_iteration > initial_iterations:
                this_iteration = (
                    training_iteration - initial_iterations
                ) % extend_every_n_iterations
                if this_iteration >= (2.0 / 3.0) * extend_every_n_iterations:

                    self.losses.variant_reconstruction_mode = "huber"

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class Scheduler:
    def __init__(self, settings, scene, renderer, trainer, data_handler):

        self.timeline = Timeline(settings, renderer, data_handler)

        self.batch_composition = BatchComposition(settings, self.timeline, data_handler)
        self.learning_rate_scheduler = LearningRateScheduler(
            settings, trainer.get_optimizer(), self.timeline
        )
        self.zero_out_fine_deformations = ZeroOutFineDeformations(settings, self.timeline, scene)
        self.coarse_smoothness_loss = CoarseSmoothnessLoss(settings, self.timeline, trainer, scene)
        self.canonical_model_regularization_losses = CanonicalModelRegularizationLosses(
            settings, self.timeline, trainer
        )
        self.only_load_active_training_images = OnlyLoadActiveTrainingImages(
            settings, self.timeline, data_handler
        )
        self.load_current_pruning_voxel_grid = LoadCurrentPruningVoxelGrid(
            settings, self.timeline, scene, data_handler
        )
        self.variants_reconstruction_losses = VariantsReconstructionLosses(
            settings, self.timeline, trainer
        )

    def state_dict(self):
        return {
            "timeline": self.timeline.state_dict(),
            "batch_composition": self.batch_composition.state_dict(),
            "learning_rate_scheduler": self.learning_rate_scheduler.state_dict(),
            "zero_out_fine_deformations": self.zero_out_fine_deformations.state_dict(),
            "coarse_smoothness_loss": self.coarse_smoothness_loss.state_dict(),
            "canonical_model_regularization_losses": self.canonical_model_regularization_losses.state_dict(),
            "only_load_active_training_images": self.only_load_active_training_images.state_dict(),
            "load_current_pruning_voxel_grid": self.load_current_pruning_voxel_grid.state_dict(),
            "variants_reconstruction_losses": self.variants_reconstruction_losses.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.timeline.load_state_dict(state_dict["timeline"])
        self.batch_composition.load_state_dict(state_dict["batch_composition"])
        self.learning_rate_scheduler.load_state_dict(state_dict["learning_rate_scheduler"])
        self.zero_out_fine_deformations.load_state_dict(state_dict["zero_out_fine_deformations"])
        self.coarse_smoothness_loss.load_state_dict(state_dict["coarse_smoothness_loss"])
        self.canonical_model_regularization_losses.load_state_dict(
            state_dict["canonical_model_regularization_losses"]
        )
        self.only_load_active_training_images.load_state_dict(
            state_dict["only_load_active_training_images"]
        )
        self.load_current_pruning_voxel_grid.load_state_dict(
            state_dict["load_current_pruning_voxel_grid"]
        )
        self.variants_reconstruction_losses.load_state_dict(
            state_dict["variants_reconstruction_losses"]
        )

    def schedule(self, training_iteration, log):

        self.timeline.update(training_iteration, log=log)

        self.batch_composition.schedule(training_iteration)
        self.only_load_active_training_images.schedule()
        self.learning_rate_scheduler.schedule(training_iteration, self.batch_composition)
        self.coarse_smoothness_loss.schedule(training_iteration, self.learning_rate_scheduler)
        self.canonical_model_regularization_losses.schedule(
            training_iteration, self.learning_rate_scheduler
        )
        self.zero_out_fine_deformations.schedule(training_iteration, self.learning_rate_scheduler)
        self.load_current_pruning_voxel_grid.schedule(training_iteration)
        self.variants_reconstruction_losses.schedule(training_iteration)

    def reset_for_rendering(self):
        self.zero_out_fine_deformations.reset_for_rendering()
