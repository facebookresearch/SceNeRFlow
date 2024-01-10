# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import coloredlogs
from data_handler import DataHandler
from multi_gpu import multi_gpu_cleanup, multi_gpu_setup, multi_gpu_train
from path_renderer import PathRenderer
from renderer import Renderer
from scene import Scene
from scheduler import Scheduler
from settings import config_parser
from state_loader_saver import StateLoaderSaver
from trainer import Trainer
from utils import (
    check_for_early_interruption,
    fix_random_number_generators,
    overwrite_settings_for_dnerf,
    overwrite_settings_for_nrnerf,
    overwrite_settings_for_pref,
)
from visualizer import Visualizer

LOGGER = logging.getLogger(__name__)


def get_end_of_timestep(training_iteration, settings, scheduler):
    return (
        settings.optimization_mode == "per_timestep"
        and settings.tracking_mode == "temporal"
        and (
            (training_iteration + 1) == scheduler.timeline.initial_iterations
            or (
                training_iteration + 1 > scheduler.timeline.initial_iterations
                and (training_iteration + 1 - scheduler.timeline.initial_iterations)
                % scheduler.timeline.extend_every_n_iterations
                == 0
            )
        )
    )


def get_do_render(training_iteration, end_of_timestep, last_iteration, settings):
    return (
        end_of_timestep
        or last_iteration
        or (
            (settings.optimization_mode != "per_timestep" or settings.tracking_mode != "temporal")
            and (training_iteration >= 0 and (training_iteration + 1) % settings.i_video == 0)
        )
    )


def train(rank=0, settings=None, world_size=1, port=None):

    if settings.multi_gpu:
        multi_gpu_setup(rank, world_size, port)

    if settings.debug:
        fix_random_number_generators(seed=rank)
        import torch

        torch.autograd.set_detect_anomaly(True)

    state_loader_saver = StateLoaderSaver(settings, rank)
    state_loader_saver.backup_files(settings)

    try:
        data_handler = DataHandler(settings)
    except RuntimeError as e:
        if "pref index out of bounds" in str(e):
            return
        else:
            raise e
    if settings.always_load_full_dataset:
        num_total_rays_to_precompute = int(1e9)
        data_handler.load_training_set(
            factor=settings.factor,
            num_total_rays_to_precompute=num_total_rays_to_precompute,
            foreground_focused=True,
            also_load_top_left_corner_and_four_courners=True,
        )
    else:
        data_handler.load_training_set(
            factor=16,
            num_pixels_per_image=5,  # some dummy value
            also_load_top_left_corner_and_four_courners=True,
        )  # the top_left_corner (Timeline in scheduler) and four_corners (determine_nerf_volume_extent in state_loader_saver) subsets need to be loaded

    scene = Scene(settings, data_handler).cuda()  # incl. time line
    renderer = Renderer(settings).cuda()

    trainer = Trainer(settings, scene, renderer, world_size)

    scheduler = Scheduler(settings, scene, renderer, trainer, data_handler)

    state_loader_saver.initialize_parameters(
        scene, renderer, scheduler, trainer, data_handler
    )  # incl. pos_min/pos_max

    # if settings.debug and rank == 0:
    #    data_handler.visualize_images_in_3D(state_loader_saver.get_results_folder())

    log = None

    first_iteration = True
    starting_iteration = state_loader_saver.get_last_stored_training_iteration()
    num_training_iterations = scheduler.timeline.get_num_training_iterations(
        settings.num_iterations
    )

    # take care of potentially interrupted renderings
    training_iteration = starting_iteration
    last_iteration = training_iteration == num_training_iterations - 1
    end_of_timestep = get_end_of_timestep(training_iteration, settings, scheduler)
    only_render = get_do_render(training_iteration, end_of_timestep, last_iteration, settings)
    if not only_render:
        starting_iteration += 1

    from tqdm import trange

    for training_iteration in trange(starting_iteration, num_training_iterations):

        # training
        scheduler.schedule(training_iteration, log)

        if not only_render:
            batch = data_handler.get_batch(batch_size=settings.batch_size // world_size)

            trainer.zero_grad()

            log = trainer.losses_and_backward_with_virtual_batches(
                batch, scene, renderer, scheduler, training_iteration
            )

            trainer.step()

        scheduler.reset_for_rendering()

        last_iteration = training_iteration == num_training_iterations - 1
        end_of_timestep = get_end_of_timestep(training_iteration, settings, scheduler)
        do_render = get_do_render(training_iteration, end_of_timestep, last_iteration, settings)

        # checkpoint
        if (
            (training_iteration + 1) % settings.save_temporary_checkpoint_every == 0
            or last_iteration
        ) and not first_iteration:
            state_loader_saver.save(
                training_iteration,
                scene,
                renderer,
                scheduler,
                trainer,
                force_save_in_stable=last_iteration,
            )

        if (
            settings.save_per_timestep
            and (end_of_timestep or last_iteration)
            and not first_iteration
        ):
            timestep = float(data_handler.precomputed["timesteps"].numpy()[0])
            state_loader_saver.save_for_only_test(
                timestep, scene, renderer, stable_storage=not settings.save_per_timestep_in_scratch
            )

        # rendering/visualization
        if do_render:
            path_rendering = PathRenderer(data_handler, rank, world_size)
            test_cameras = data_handler.get_test_cameras_for_rendering(
                factor=settings.factor if settings.render_factor == 0 else settings.render_factor
            )

            also_render_coarse = True
            if also_render_coarse and scene.deformation_model.coarse_and_fine:
                scheduler.zero_out_fine_deformations.fine_deformations(zero_out=True)
                path_rendering.render_and_store(
                    state_loader_saver,
                    also_store_images=True,
                    output_name=state_loader_saver.get_experiment_name()
                    + "_"
                    + str(training_iteration).zfill(8)
                    + "_coarse",
                    scene=scene,
                    renderer=renderer,
                    **test_cameras
                )
                scheduler.zero_out_fine_deformations.fine_deformations(zero_out=False)

            path_rendering.render_and_store(
                state_loader_saver,
                also_store_images=True,
                output_name=state_loader_saver.get_experiment_name()
                + "_"
                + str(training_iteration).zfill(8),
                scene=scene,
                renderer=renderer,
                **test_cameras
            )

            # if "backgrounds" in test_cameras:
            #    del test_cameras["backgrounds"]
            #    path_rendering.render_and_store(state_loader_saver, output_name=str(training_iteration).zfill(8) + "_nobackground", \
            #        scene=scene, renderer=renderer, **test_cameras)

            if settings.use_visualizer:
                visualizer = Visualizer(settings, data_handler, rank, world_size)
                test_cameras = data_handler.get_test_cameras_for_rendering(factor=8)
                visualizer.render_and_store(
                    state_loader_saver,
                    output_name=state_loader_saver.get_experiment_name()
                    + "_"
                    + str(training_iteration).zfill(8),
                    scene=scene,
                    renderer=renderer,
                    **test_cameras
                )

        # logging
        if (training_iteration + 1) % settings.i_print == 0:
            state_loader_saver.print_log(training_iteration, log)

        # interruption
        if training_iteration % 3000 == 0:
            try:
                check_for_early_interruption(state_loader_saver)
            except RuntimeError:
                break

        first_iteration = False
        only_render = False  # only the first iteration can be an "only render" iteration

    if settings.multi_gpu:
        multi_gpu_cleanup(rank)


if __name__ == "__main__":

    settings = config_parser().parse_args()
    if settings.do_pref:
        settings = overwrite_settings_for_pref(settings)
    if settings.do_nrnerf:
        settings = overwrite_settings_for_nrnerf(settings)
    if settings.do_dnerf:
        settings = overwrite_settings_for_dnerf(settings)

    logging_level = logging.DEBUG if settings.debug else logging.INFO
    coloredlogs.install(level=logging_level, fmt="%(name)s[%(process)d] %(levelname)s %(message)s")
    logging.basicConfig(level=logging_level)

    if settings.multi_gpu:
        multi_gpu_train(settings)
    else:
        train(settings=settings)
