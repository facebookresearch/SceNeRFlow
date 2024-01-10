# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
python evaluation.py ./results/toy_example
To produce masked scores, use:
python evaluation.py ./results/toy_example --masked
"""

import logging
import os
import sys

import coloredlogs
import imageio.v2 as imageio
import numpy as np
import torch

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def create_folder(folder):
    import pathlib

    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


# EVALUATION: NOVEL VIEWS


def _evaluate_image(
    generated,
    groundtruth,
    imageid,
    naive_error_folder=None,
    ssim_error_folder=None,
    perceptual_metric=None,
    mask=None,
):
    def visualize_with_jet_color_scheme(image):
        from matplotlib import cm

        color_mapping = np.array([cm.jet(i)[:3] for i in range(256)])
        max_value = 1.0
        min_value = 0.0
        intermediate = (
            np.clip(image, a_max=max_value, a_min=min_value) / max_value
        )  # cut off above max_value. result is normalized to [0,1]
        intermediate = (255.0 * intermediate).astype("uint8")  # now contains int in [0,255]
        original_shape = intermediate.shape
        intermediate = color_mapping[intermediate.flatten()]
        intermediate = intermediate.reshape(original_shape + (3,))
        return intermediate

    # mask
    if mask is not None:
        generated[mask] = 0.0
        groundtruth[mask] = 0.0

    # PSNR
    mse = np.mean((groundtruth - generated) ** 2)
    psnr = -10.0 * np.log10(mse)

    # SSIM
    # https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity
    from skimage.metrics import structural_similarity as ssim

    create_ssim_error_map = ssim_error_folder is not None
    returned = ssim(
        groundtruth,
        generated,
        data_range=1.0,
        multichannel=True,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        full=create_ssim_error_map,
    )
    if create_ssim_error_map:
        ssim_error, ssim_error_image = returned
    else:
        ssim_error = returned

    # perceptual metric
    if perceptual_metric is None:
        lpips = 1.0
    else:

        def numpy_to_pytorch(np_image):
            torch_image = (
                2 * torch.from_numpy(np_image) - 1
            )  # height x width x 3. must be in [-1,+1]
            torch_image = torch_image.permute(2, 0, 1)  # 3 x height x width
            return torch_image.unsqueeze(0)  # 1 x 3 x height x width

        lpips = perceptual_metric.forward(
            numpy_to_pytorch(groundtruth), numpy_to_pytorch(generated)
        )
        lpips = float(lpips.detach().reshape(1).numpy()[0])

    scores = {"psnr": float(psnr), "ssim": float(ssim_error), "lpips": float(lpips)}

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    if naive_error_folder is not None:
        # MSE-style
        error = np.linalg.norm(groundtruth - generated, axis=-1) / np.sqrt(
            1 + 1 + 1
        )  # height x width
        error *= 10.0  # exaggarate error
        error = np.clip(error, 0.0, 1.0)
        error = to8b(
            visualize_with_jet_color_scheme(error)
        )  # height x width x 3. int values in [0,255]
        filename = os.path.join(naive_error_folder, "error_{:03d}.png".format(imageid))
        imageio.imwrite(filename, error)

    if ssim_error_folder is not None:
        # SSIM
        filename = os.path.join(ssim_error_folder, "error_{:03d}.png".format(imageid))
        ssim_error_image = to8b(
            visualize_with_jet_color_scheme(1.0 - np.mean(ssim_error_image, axis=-1))
        )
        imageio.imwrite(filename, ssim_error_image)

    return scores


def quantitative_evaluation_novel_views(results_folder, use_masks):

    output_name = "novel_view_eval"

    only_render_if_file_does_not_exist = True
    store_masks = True  # whether to write the masks used for masked evaluation into files

    background_baseline = (
        False  # whether to compute the static background image baseline from the paper
    )
    create_error_maps = False  # whether to write error maps into files
    factor = 2  # downsampling factor of image resolution

    # masked evaluation. needs to be customized per scene for decent results.
    background_threshold = 10.0 / 255.0  # color. consider 3.0, 10.0, 15.0
    diff_background_dilation_kernel_size = 11  # in pixels. paper uses {7, 11, 13, 15}
    diff_foreground_dilation_kernel_size = 71  # in pixels
    std_threshold = 4.0 / 255.0  # color. paper uses {2.5, 3.0, 4.0, 5.0}
    std_background_dilation_kernel_size = 11  # in pixels. paper uses {7, 11, 13}
    std_foreground_dilation_kernel_size = 71  # in pixels

    output_folder = os.path.join(results_folder, "4_outputs", output_name)

    create_folder(output_folder)

    if background_baseline:
        output_name += "_background"
    use_latest_checkpoint = False

    if create_error_maps:
        naive_error_folder = os.path.join(output_folder, "naive_errors")
        create_folder(naive_error_folder)
        ssim_error_folder = os.path.join(output_folder, "ssim_errors")
        create_folder(ssim_error_folder)
    else:
        naive_error_folder = None
        ssim_error_folder = None

    # code_folder = os.path.join(results_folder, "2_backup/")
    # sys.path = [code_folder] + sys.path

    from settings import config_parser

    settings, _ = config_parser(
        config_file=os.path.join(results_folder, "settings.txt")
    ).parse_known_args()
    do_pref = settings.do_pref
    if do_pref:
        from utils import overwrite_settings_for_pref

        settings = overwrite_settings_for_pref(settings)
        use_latest_checkpoint = True
    do_nrnerf = settings.do_nrnerf
    if do_nrnerf:
        from utils import overwrite_settings_for_nrnerf

        settings = overwrite_settings_for_nrnerf(settings)
        use_latest_checkpoint = True
    do_dnerf = settings.do_dnerf
    if do_dnerf:
        use_latest_checkpoint = True
    from state_loader_saver import StateLoaderSaver

    rank = 0
    state_loader_saver = StateLoaderSaver(settings, rank)
    from data_handler import DataHandler

    data_handler = DataHandler(settings)
    data_handler.load_training_set(
        factor=16,
        num_pixels_per_image=5,
        also_load_top_left_corner_and_four_courners=False,
    )
    data_loader = data_handler.data_loader
    from scene import Scene

    scene = Scene(settings, data_handler).cuda()
    from renderer import Renderer

    renderer = Renderer(settings).cuda()

    if use_latest_checkpoint:
        filename = "latest.pth"
    else:
        filename = "timestep_0.0.pth"
    verify_checkpoint_file = os.path.join(state_loader_saver.get_checkpoint_folder(), filename)
    if os.path.exists(verify_checkpoint_file):
        checkpoint_folder = state_loader_saver.get_checkpoint_folder()
    else:
        checkpoint_folder = state_loader_saver.get_temporary_checkpoint_folder()

    checkpoint_file = os.path.join(checkpoint_folder, filename)
    checkpoint = torch.load(checkpoint_file)
    scene.load_state_dict(checkpoint["scene"])
    renderer.load_state_dict(checkpoint["renderer"])

    class _load_checkpoint_for_timestep:
        def __init__(self):
            self.loaded_timestep = 0.0

        def load(self, timestep):
            if timestep == self.loaded_timestep or use_latest_checkpoint:
                return

            self.loaded_timestep = timestep

            checkpoint_file = os.path.join(checkpoint_folder, "timestep_" + str(timestep) + ".pth")
            if not os.path.exists(checkpoint_file):
                raise FileNotFoundError
            checkpoint = torch.load(checkpoint_file)
            try:
                timevariant_canonical_model = (
                    scene.canonical_model.variant in ["snfa", "snfag"]
                    or scene.canonical_model.brightness_variability > 0.0
                )
            except:
                try:
                    timevariant_canonical_model = (
                        scene.canonical_model.use_time_conditioning
                        or scene.canonical_model.brightness_variability > 0.0
                    )
                except:
                    timevariant_canonical_model = scene.canonical_model.brightness_variability > 0.0
            if timevariant_canonical_model or timestep == 0.0:
                scene.load_state_dict(checkpoint["scene"])
            else:
                if scene.deformation_model is not None:
                    scene.deformation_model.load_state_dict(checkpoint["deformation_model"])

    hacky_checkpoint_loading = _load_checkpoint_for_timestep()

    try:
        import lpips

        perceptual_metric = lpips.LPIPS(net="alex")
    except:
        LOGGER.warning(
            "Perceptual LPIPS metric not found. Will skip LPIPS. Please see the README for installation instructions."
        )
        perceptual_metric = None

    rendered_folder = os.path.join(output_folder, "rendered")
    create_folder(rendered_folder)

    if background_baseline or use_masks:
        all_background_images = data_loader.load_background_images(
            factor=factor
        )  # N x H x W x 3. ordered by exintrinid

    scores = {}
    all_rgbs = []

    all_test_imageids = data_loader.get_test_imageids()
    from tqdm import tqdm

    for test_imageid in tqdm(all_test_imageids):
        groundtruth = data_loader.load_images(
            factor=factor, imageids=np.array([test_imageid]).astype(np.int32)
        )[0]["everything"][
            0
        ]  # H x W x 3

        if background_baseline:
            exintrinid = data_loader.get_exintrinids(test_imageid).item()
            rendered_image = all_background_images[exintrinid]
        else:
            if use_masks:
                exintrinid = data_loader.get_exintrinids(test_imageid).item()
                background_image = all_background_images[exintrinid]

                def process_mask(
                    mask, background_dilation_kernel_size, foreground_dilation_kernel_size
                ):
                    mask = torch.unsqueeze(
                        torch.unsqueeze(mask, dim=0), dim=0
                    )  # 1 x 1 x height x width
                    kernel_size = background_dilation_kernel_size
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # needed for integer padding
                    padding = (kernel_size - 1) // 2
                    mask = (
                        torch.nn.functional.max_pool2d(
                            mask.float(),
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                            dilation=1,
                            ceil_mode=True,
                            return_indices=False,
                        )
                        == 1.0
                    )  # dilate background
                    kernel_size = foreground_dilation_kernel_size
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # needed for integer padding
                    padding = (kernel_size - 1) // 2
                    mask = (
                        torch.nn.functional.max_pool2d(
                            1.0 - mask.float(),
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                            dilation=1,
                            ceil_mode=True,
                            return_indices=False,
                        )
                        == 0.0
                    )  # dilate foreground
                    mask = mask[0, 0]  # height x width
                    return mask

                difference_image = torch.mean(
                    torch.abs(torch.from_numpy(groundtruth) - background_image), axis=-1
                )  # height x width
                mask1 = difference_image < background_threshold
                mask1 = process_mask(
                    mask1,
                    diff_background_dilation_kernel_size,
                    diff_foreground_dilation_kernel_size,
                )

                difference_image = torch.std(
                    torch.from_numpy(groundtruth) / (background_image + 0.001), axis=-1
                )  # height x width
                mask2 = difference_image < std_threshold
                mask2 = process_mask(
                    mask2, std_background_dilation_kernel_size, std_foreground_dilation_kernel_size
                )

                mask = torch.logical_or(mask1, mask2)

                if store_masks:
                    mask_folder = os.path.join(output_folder, "masks")
                    create_folder(mask_folder)
                    mask_to_save = (255 * mask.cpu().numpy().astype(np.int32)).astype(np.uint8)
                    imageio.imsave(
                        os.path.join(mask_folder, str(test_imageid).zfill(8) + ".jpg"),
                        mask_to_save,
                        quality=90,
                    )

            imageids = np.array([test_imageid]).astype(np.int32)
            test_cameras = {}
            test_cameras["timesteps"] = data_loader.get_timesteps(imageids).cpu()
            extrinids = data_loader.get_extrinids(imageids)
            test_cameras["extrins"] = data_loader.get_extrinsics(extrinids=extrinids)
            intrinids = data_loader.get_intrinids(imageids)
            test_cameras["intrins"] = data_loader.get_intrinsics(intrinids=intrinids, factor=factor)
            test_cameras["rgb"] = torch.from_numpy(groundtruth).unsqueeze(0)
            if data_loader.has_background_images():
                exintrinids = data_loader.get_exintrinids(imageids)
                test_cameras["backgrounds"] = torch.from_numpy(
                    data_loader.load_background_images(factor=factor, exintrinids=exintrinids)
                )

            output_name = str(test_imageid).zfill(8)
            from path_renderer import PathRenderer

            world_size = 1
            path_rendering = PathRenderer(data_handler, rank, world_size)
            try:
                path_rendering.render_and_store(
                    state_loader_saver,
                    output_name,
                    output_folder=rendered_folder,
                    scene=scene,
                    renderer=renderer,
                    hacky_checkpoint_loading=hacky_checkpoint_loading,
                    also_store_images=True,
                    only_render_if_file_does_not_exist=only_render_if_file_does_not_exist,
                    **test_cameras
                )
            except AssertionError:
                pass  # hacky workaround to make sure hacky_checkpoint_loading is only used intentionally
            except FileNotFoundError:
                continue

            file_name = os.path.join(rendered_folder, output_name + "_rgb.mp4")
            rendered_image = imageio.imread(file_name + "_00000.jpg")
            all_rgbs.append(rendered_image.copy())
            rendered_image = rendered_image.astype(np.float32) / 255.0

        these_scores = _evaluate_image(
            rendered_image,
            groundtruth,
            test_imageid,
            naive_error_folder=naive_error_folder,
            ssim_error_folder=ssim_error_folder,
            perceptual_metric=perceptual_metric,
            mask=None,
        )

        LOGGER.info(these_scores)

        scores[int(test_imageid)] = these_scores

        if use_masks:
            these_masked_scores = _evaluate_image(
                rendered_image,
                groundtruth,
                test_imageid,
                naive_error_folder=naive_error_folder,
                ssim_error_folder=ssim_error_folder,
                perceptual_metric=perceptual_metric,
                mask=mask,
            )

            these_masked_scores = {
                "masked_" + key: value for key, value in these_masked_scores.items()
            }

            LOGGER.info(these_masked_scores)

            scores[int(test_imageid)].update(these_masked_scores)

    averaged_scores = {}
    averaged_scores["average_psnr"] = np.mean([score["psnr"] for score in scores.values()])
    averaged_scores["average_ssim"] = np.mean([score["ssim"] for score in scores.values()])
    averaged_scores["average_lpips"] = np.mean([score["lpips"] for score in scores.values()])

    if use_masks:
        averaged_scores["average_masked_psnr"] = np.mean(
            [score["masked_psnr"] for score in scores.values()]
        )
        averaged_scores["average_masked_ssim"] = np.mean(
            [score["masked_ssim"] for score in scores.values()]
        )
        averaged_scores["average_masked_lpips"] = np.mean(
            [score["masked_lpips"] for score in scores.values()]
        )

    scores.update(averaged_scores)

    import json

    with open(os.path.join(output_folder, "scores.json"), "w", encoding="utf-8") as json_file:
        json.dump(scores, json_file, ensure_ascii=False, indent=4)

    if len(all_rgbs) > 0:
        imageio.mimwrite(
            os.path.join(output_folder, "video.mp4"), np.stack(all_rgbs, axis=0), fps=25, quality=10
        )

    # sys.path.remove(code_folder)


if __name__ == "__main__":

    # logging_level = logging.DEBUG
    logging_level = logging.INFO
    coloredlogs.install(level=logging_level, fmt="%(name)s[%(process)d] %(levelname)s %(message)s")
    logging.basicConfig(level=logging_level)

    results_folder = sys.argv[1]

    use_masks = len(sys.argv) > 2
    if use_masks:
        LOGGER.info("Unmasked and masked evaluation...")
    else:
        LOGGER.info("Only unmasked evaluation...")
    quantitative_evaluation_novel_views(results_folder, use_masks)
