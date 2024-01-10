# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import configargparse


def config_parser(config_file=None):

    if config_file is None:
        config_file = "./configs/default.txt"

    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--config", default=config_file, is_config_file=True, help="config file path"
    )
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument(
        "--basedir", type=str, default="./logs/", help="where to store ckpts and logs"
    )
    parser.add_argument(
        "--temporary_basedir", type=str, default=None, help="where to store ckpts and logs"
    )
    parser.add_argument("--datadir", type=str, default=None, help="input data directory")
    parser.add_argument(
        "--allow_scratch_datadir_copy",
        action="store_true",
        help="only take random rays from 1 image at a time",
    )

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help="layers in network")
    parser.add_argument("--netwidth", type=int, default=256, help="channels per layer")
    parser.add_argument("--netdepth_fine", type=int, default=8, help="layers in fine network")
    parser.add_argument(
        "--netwidth_fine", type=int, default=256, help="channels per layer in fine network"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument(
        "--points_per_chunk",
        type=int,
        default=1e7,
        help="number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_batching", action="store_true", help="only take random rays from 1 image at a time"
    )
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument("--tracking_mode", type=str, default="plain", help="plain, temporal")
    parser.add_argument("--reconstruction_loss_type", type=str, default="L1", help="L1, L2")
    parser.add_argument(
        "--smooth_deformations_type",
        type=str,
        default="finite",
        help="finite, divergence, jacobian",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=200000, help="number of training iterations"
    )
    parser.add_argument(
        "--learning_rate_decay_autodecoding_fraction",
        type=float,
        default=1e-2,
        help="what fraction of the learning rate to reduce to",
    )
    parser.add_argument(
        "--learning_rate_decay_autodecoding_iterations",
        type=int,
        default=0,
        help='number of iterations to reduce learning rate by "fraction"',
    )
    parser.add_argument(
        "--learning_rate_decay_mlp_fraction",
        type=float,
        default=1e-2,
        help="what fraction of the learning rate to reduce to",
    )
    parser.add_argument(
        "--learning_rate_decay_mlp_iterations",
        type=int,
        default=0,
        help="number of iterations to reduce learning rate",
    )
    parser.add_argument(
        "--activation_function",
        type=str,
        default="ReLU",
        help="ReLU, Exponential, Sine, Squareplus",
    )

    parser.add_argument(
        "--use_visualizer", action="store_true", help="auto-decoded latent codes or raw time"
    )
    parser.add_argument(
        "--test_cameras", nargs="+", type=str, help="extrinsic names of test cameras", default=[]
    )

    # rendering options
    parser.add_argument(
        "--num_points_per_ray", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance", type=int, default=0, help="number of additional fine samples per ray"
    )
    parser.add_argument(
        "--perturb", type=float, default=1.0, help="set to 0. for no jitter, 1. for jitter"
    )
    parser.add_argument(
        "--use_viewdirs", action="store_true", help="use full 5D input instead of 3D"
    )
    parser.add_argument(
        "--i_embed", type=int, default=0, help="set 0 for default positional encoding, -1 for none"
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help="std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )
    parser.add_argument(
        "--use_background",
        action="store_true",
        help="use static background images when composing rays",
    )
    parser.add_argument(
        "--brightness_variability",
        type=float,
        default=0.0,
        help="maximum allowed change in learned brightness. up to 1.0 makes sense. 0.0 turns it off.",
    )
    parser.add_argument(
        "--variant", type=str, default="snf", help="options: llff / blender / deepvoxels"
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    parser.add_argument(
        "--color_calibration_mode",
        type=str,
        default="none",
        help="options: none / full_matrix / neural_volumes",
    )

    # training options
    parser.add_argument(
        "--weight_smooth_deformations",
        type=float,
        default=0.0,
        help="weight for regularization loss",
    )
    parser.add_argument(
        "--weight_coarse_smooth_deformations",
        type=float,
        default=0.0,
        help="weight for regularization loss",
    )
    parser.add_argument(
        "--weight_fine_smooth_deformations",
        type=float,
        default=0.0,
        help="weight for regularization loss",
    )
    parser.add_argument(
        "--weight_parameter_regularization",
        type=float,
        default=0.0,
        help="weight for weight decay in AdamW",
    )
    parser.add_argument(
        "--weight_background_loss", type=float, default=0.0, help="weight for background loss"
    )
    parser.add_argument(
        "--weight_brightness_change_regularization",
        type=float,
        default=0.0,
        help="weight for brightness change regularization",
    )
    parser.add_argument(
        "--weight_hard_surface_loss",
        type=float,
        default=0.0,
        help="weight for brightness change regularization",
    )
    parser.add_argument(
        "--weight_small_fine_offsets_loss",
        type=float,
        default=0.0,
        help="weight for brightness change regularization",
    )
    parser.add_argument(
        "--weight_similar_coarse_and_total_offsets_loss",
        type=float,
        default=0.0,
        help="weight for brightness change regularization",
    )
    parser.add_argument(
        "--weight_per_frequency_regularization",
        type=float,
        default=0.0,
        help="weight for brightness change regularization",
    )

    parser.add_argument(
        "--coarse_and_fine", action="store_true", help="auto-decoded latent codes or raw time"
    )
    parser.add_argument(
        "--fine_range",
        type=float,
        default=0.1,
        help="hard restriction on the range of the fine deformation model in normalized space",
    )
    parser.add_argument(
        "--deformation_per_timestep_decay_rate",
        type=float,
        default=0.1,
        help="weight for brightness change regularization",
    )
    parser.add_argument(
        "--slow_canonical_per_timestep_learning_rate",
        type=float,
        default=1e-5,
        help="weight for brightness change regularization",
    )
    parser.add_argument(
        "--fix_coarse_after_a_while",
        action="store_true",
        help="auto-decoded latent codes or raw time",
    )
    parser.add_argument(
        "--let_canonical_vary_at_last",
        action="store_true",
        help="auto-decoded latent codes or raw time",
    )
    parser.add_argument(
        "--let_only_brightness_vary",
        action="store_true",
        help="auto-decoded latent codes or raw time",
    )
    parser.add_argument(
        "--keep_coarse_mlp_constant",
        action="store_true",
        help="auto-decoded latent codes or raw time",
    )
    parser.add_argument(
        "--coarse_parametrization",
        type=str,
        default="MLP",
        help="auto-decoded latent codes or raw time",
    )
    parser.add_argument(
        "--use_global_transform", action="store_true", help="auto-decoded latent codes or raw time"
    )
    parser.add_argument(
        "--do_vignetting_correction",
        action="store_true",
        help="auto-decoded latent codes or raw time",
    )
    parser.add_argument(
        "--coarse_mlp_weight_decay", type=float, default=1e-2, help="weight for background loss"
    )
    parser.add_argument(
        "--coarse_mlp_skip_connections",
        type=int,
        default=0,
        help="downsample factor for LLFF images",
    )
    parser.add_argument(
        "--smoothness_robustness_threshold",
        type=float,
        default=1e-2,
        help="weight for background loss",
    )

    parser.add_argument(
        "--use_half_precision",
        action="store_true",
        help="nerfies deformation parametrization instead of offsets field",
    )
    parser.add_argument(
        "--do_zero_out",
        action="store_true",
        help="nerfies deformation parametrization instead of offsets field",
    )

    parser.add_argument(
        "--use_pruning",
        action="store_true",
        help="nerfies deformation parametrization instead of offsets field",
    )
    parser.add_argument(
        "--voxel_grid_size", type=int, default=128, help="downsample factor for LLFF images"
    )
    parser.add_argument(
        "--no_pruning_probability",
        type=float,
        default=0.0,
        help="hard restriction on the range of the fine deformation model in normalized space",
    )

    parser.add_argument(
        "--per_frequency_training_mode",
        type=str,
        default="only_last",
        help="options: up_to_last / only_last / all_at_once",
    )
    parser.add_argument(
        "--optimization_mode", type=str, default="per_timestep", help="options: per_timestep / all"
    )
    parser.add_argument("--do_ngp_mip_nerf", action="store_true", help="submit to slurm")
    parser.add_argument("--do_pref", action="store_true", help="submit to slurm")
    parser.add_argument(
        "--pref_tau_window",
        type=int,
        default=3,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )
    parser.add_argument(
        "--pref_dataset_index",
        type=int,
        default=-1,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )
    parser.add_argument("--do_nrnerf", action="store_true", help="submit to slurm")
    parser.add_argument("--do_dnerf", action="store_true", help="submit to slurm")

    # dataset options
    parser.add_argument(
        "--dataset_type", type=str, default="blender", help="options: llff / blender / deepvoxels"
    )
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    # deformation options
    parser.add_argument(
        "--use_temporal_latent_codes",
        action="store_true",
        help="auto-decoded latent codes or raw time",
    )
    parser.add_argument("--pure_mlp_bending", action="store_true", help="pure MLP bending network")
    parser.add_argument(
        "--use_time_conditioning",
        action="store_true",
        help="time-condition the canonical model for deformations.",
    )
    parser.add_argument(
        "--use_nerfies_se3",
        action="store_true",
        help="nerfies deformation parametrization instead of offsets field",
    )

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, help="downsample factor for LLFF images")
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help="do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--disparity_sampling",
        action="store_true",
        help="sampling linearly in disparity instead of depth",
    )
    parser.add_argument("--spherify", action="store_true", help="set for spherical 360 scenes")
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    # logging/saving options
    parser.add_argument(
        "--i_print", type=int, default=100, help="frequency of console printout and metric loggin"
    )
    parser.add_argument(
        "--i_img", type=int, default=2500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--save_temporary_checkpoint_every",
        type=int,
        default=2500,
        help="frequency of weight ckpt saving",
    )
    parser.add_argument(
        "--save_intermediate_checkpoint_every",
        type=int,
        default=10000,
        help="frequency of weight ckpt saving",
    )
    parser.add_argument(
        "--save_checkpoint_every", type=int, default=2500, help="frequency of weight ckpt saving"
    )
    parser.add_argument(
        "--save_per_timestep", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument(
        "--save_per_timestep_in_scratch", action="store_true", help="set for spherical 360 scenes"
    )
    parser.add_argument("--i_testset", type=int, default=50000, help="frequency of testset saving")
    parser.add_argument(
        "--i_video", type=int, default=50000, help="frequency of render_poses video saving"
    )

    # backbone network
    parser.add_argument("--backbone", type=str, default="mlp", help="backbone: mlp / ngp")
    parser.add_argument(
        "--prefer_cutlass_over_fullyfused_mlp",
        action="store_true",
        help="cutlass for gpu20, fullyfused for gpu22",
    )

    parser.add_argument("--slurm", action="store_true", help="submit to slurm")
    parser.add_argument(
        "--time_cpu_ram_cluster_gpu",
        type=str,
        default="1:00:00 16 200 gpu20 gpu:1",
        help="settings for slurm",
    )

    parser.add_argument("--multi_gpu", action="store_true", help="whether to use multiple GPUs")
    parser.add_argument("--debug", action="store_true", help="debug flag")
    parser.add_argument(
        "--always_load_full_dataset",
        action="store_true",
        help="auto-decoded latent codes or raw time",
    )

    return parser
