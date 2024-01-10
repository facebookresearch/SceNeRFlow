# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch

LOGGER = logging.getLogger(__name__)


def fix_random_number_generators(seed=None):
    if seed is None:
        seed = 0
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)


def check_for_early_interruption(state_loader_saver):

    import os

    results_folder = state_loader_saver.get_results_folder()

    # subfolders
    folders_to_check = [
        os.path.join(results_folder, folder) for folder in os.listdir(results_folder)
    ]
    folders_to_check = [folder for folder in folders_to_check if os.path.isdir(folder)]
    # root folder
    folders_to_check.append(results_folder)

    terminating_files = ["stop", "New Text Document.txt"]
    terminating_files = [file.casefold() for file in terminating_files]  # lower case normalization

    for folder in folders_to_check:
        files = os.listdir(folder)
        files = [file.casefold() for file in files]

        if any(terminating_file in files for terminating_file in terminating_files):
            LOGGER.warning("Shutting down early.")
            raise RuntimeError


def project_to_correct_range(values, mode, min_=None, max_=None):
    if min_ is None:
        min_ = 0.0
    if max_ is None:
        max_ = 1.0

    if mode == "clamp":
        #   -1  0  1  2
        # 1       /----
        #        /
        # 0 ____/
        return torch.clamp(values, min=min_, max=max_)

    elif mode == "sine":
        # sine, but shifted and scaled such that it's close to the identity on [0,1]
        #   -1  0  1  2
        # 1  -    --
        #     \  /  \
        # 0    --    -
        if min_ != 0.0 or max_ != 1.0:
            values = (values - min_) / (max_ - min_)
        # values = (torch.sin(values * np.pi - np.pi/2.0) + 1.0) / 2.0
        values = (1.0 - torch.cos(values * torch.pi)) / 2.0
        if min_ != 0.0 or max_ != 1.0:
            values = (max_ - min_) * values + min_
        return values

    elif mode == "zick_zack":
        # identity function on [0,1]. reflecting boundary. linear.
        #   -1  0  1  2
        # 1  \    /\
        #     \  /  \
        # 0    \/    \
        if min_ != 0.0 or max_ != 1.0:
            values = (values - min_) / (max_ - min_)
        floor = torch.floor(values)
        fraction = values - floor  # in [0,1]
        odd_mask = floor.long() % 2 == 1
        fraction[odd_mask] = 1.0 - fraction[odd_mask]
        if min_ != 0.0 or max_ != 1.0:
            fraction = (max_ - min_) * fraction + min_
        return fraction


def szudzik(a, b):
    if a >= b:
        key = a * a + a + b
    else:
        key = a + b * b
    return key


# from FFJORD github code
def get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ..., D_y)
      x: (N, ..., D_x)
    Returns:
      The minibatch Jacobian matrix of shape (N, ..., D_y, D_x)
    """
    assert y.shape[:-1] == x.shape[:-1]
    prefix_shape = y.shape[:-1]
    y = y.view(-1, y.shape[-1])

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0]
        dy_j_dx = dy_j_dx.view(-1, x.shape[-1])
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    jac = jac.view(prefix_shape + jac.shape[-2:])
    return jac


# from FFJORD github code
def divergence_exact(inputs, outputs):
    # requires three backward passes instead of one like divergence_approx
    prefix_shape = outputs.shape[:-1]
    jac = get_minibatch_jacobian(outputs, inputs)
    diagonal = jac.view(-1, jac.shape[-1] * jac.shape[-2])[:, :: (jac.shape[-1] + 1)]
    divergence = torch.sum(diagonal, 1)

    divergence = divergence.view(prefix_shape)
    return divergence


# from FFJORD github code
def divergence_approx(inputs, outputs):
    # avoids explicitly computing the Jacobian
    e = torch.randn_like(outputs, device=outputs.get_device())
    e_dydx = torch.autograd.grad(outputs, inputs, e, create_graph=True)[0]
    e_dydx_e = e_dydx * e
    approx_tr_dydx = e_dydx_e.sum(dim=-1)
    return approx_tr_dydx


def positional_encoding(x, freq_bands=None, num_frequencies=None, include_input=None):

    # shape of x : .... x k
    # output dimensions: k + num_frequencies * 2 * k

    encoded_input = []

    if include_input is None:
        include_input = num_frequencies is not None

    if include_input:
        encoded_input.append(x)

    if num_frequencies is None:
        freq_bands = freq_bands
    else:
        freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, steps=num_frequencies)
    for frequency in freq_bands:
        encoded_input.append(torch.sin(x * frequency))
        encoded_input.append(torch.cos(x * frequency))

    encoded_input = torch.cat(encoded_input, dim=-1)

    return encoded_input


class Squareplus(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x * x + 4))


class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Scaling(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return self.factor * x


def default_relu_initialization(layer):
    with torch.no_grad():
        torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(layer.bias)


def default_sine_initialization(layer, first=False):
    import numpy as np
    with torch.no_grad():
        a = 30.0 / layer.in_features if first else np.sqrt(6.0 / layer.in_features)
        layer.weight.uniform_(-a, a)
        torch.nn.init.zeros_(layer.bias)


def zero_initialization(layer):
    with torch.no_grad():
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)


def build_pytorch_mlp_from_tinycudann(mlp_dict, half_precision=False, last_layer_zero_init=False):
    sequential = []

    activation_name = mlp_dict["network_config"]["activation"]

    def get_activation(activation_name):
        if activation_name == "ReLU":
            activation = torch.nn.ReLU()
        elif activation_name == "Exponential":
            activation = torch.nn.ELU()
        elif activation_name == "Sine":
            activation = Sine()
        elif activation_name == "Squareplus":
            activation = Squareplus()
        elif activation_name == "LeakyReLU":
            activation = torch.nn.LeakyReLU()
        else:
            raise NotImplementedError
        return activation

    activation = get_activation(activation_name)

    input_dims = mlp_dict["n_input_dims"]
    if "encoding_config" in mlp_dict:
        if mlp_dict["encoding_config"]["otype"] == "Frequency":
            input_dims = input_dims + mlp_dict["encoding_config"]["n_frequencies"] * 2 * input_dims
        elif mlp_dict["encoding_config"]["otype"] == "Limited_Frequency":
            input_dims = len(mlp_dict["encoding_config"]["frequencies"]) * 2 * input_dims
            if mlp_dict["encoding_config"]["include_input"]:
                input_dims += 3
        elif mlp_dict["encoding_config"]["otype"] == "Composite":
            if len(mlp_dict["encoding_config"]["nested"]) != 2:
                raise NotImplementedError
            if (
                mlp_dict["encoding_config"]["nested"][0]["otype"] == "Frequency"
            ):  # latent code positional encoding
                pos_enc_input_dimensions = mlp_dict["encoding_config"]["nested"][0][
                    "n_dims_to_encode"
                ]
                input_dims = (input_dims - pos_enc_input_dimensions) + (
                    pos_enc_input_dimensions
                    + mlp_dict["encoding_config"]["nested"][0]["n_frequencies"]
                    * 2
                    * pos_enc_input_dimensions
                )
            if mlp_dict["encoding_config"]["nested"][1]["otype"] == "Frequency":
                pos_enc_input_dimensions = mlp_dict["encoding_config"]["nested"][1][
                    "n_dims_to_encode"
                ]
                input_dims = (input_dims - pos_enc_input_dimensions) + (
                    pos_enc_input_dimensions
                    + mlp_dict["encoding_config"]["nested"][1]["n_frequencies"]
                    * 2
                    * pos_enc_input_dimensions
                )
            elif mlp_dict["encoding_config"]["nested"][1]["otype"] != "Identity":
                raise NotImplementedError
        else:
            raise NotImplementedError
    first_layer = torch.nn.Linear(input_dims, mlp_dict["network_config"]["n_neurons"])
    if activation_name == "Sine":
        default_sine_initialization(first_layer, first=True)
    elif activation_name in ["ReLU", "LeakyReLU"]:
        default_relu_initialization(first_layer)
    sequential.append(first_layer)
    sequential.append(activation)

    for layer in range(mlp_dict["network_config"]["n_hidden_layers"] - 1):
        layer = torch.nn.Linear(
            mlp_dict["network_config"]["n_neurons"], mlp_dict["network_config"]["n_neurons"]
        )
        if activation_name == "Sine":
            default_sine_initialization(layer)
        elif activation_name in ["ReLU", "LeakyReLU"]:
            default_relu_initialization(layer)
        sequential.append(layer)
        sequential.append(activation)

    last_layer = torch.nn.Linear(mlp_dict["network_config"]["n_neurons"], mlp_dict["n_output_dims"])
    if last_layer_zero_init:
        zero_initialization(last_layer)  # note: different from tiny cuda nn
    elif activation_name == "Sine":
        default_sine_initialization(last_layer)
    elif activation_name in ["ReLU", "LeakyReLU"]:
        default_relu_initialization(last_layer)
    sequential.append(last_layer)
    if mlp_dict["network_config"]["output_activation"] == "None":
        pass
    else:
        sequential.append(get_activation(mlp_dict["network_config"]["output_activation"]))

    mlp = torch.nn.Sequential(*sequential)
    if half_precision:  # tag:half_precision
        mlp = mlp.half()  # do not use if autocast is used instead.
    return mlp


def infill_masked(mask, masked_tensor, infill_value=None):
    if infill_value is None:
        infill_value = 0
    infilled = infill_value * torch.ones(
        mask.shape + masked_tensor.shape[1:], dtype=masked_tensor.dtype, device=masked_tensor.device
    )
    infilled[mask] = masked_tensor
    return infilled


def get_scratch_scene_folder(datadir):
    import os

    if datadir[-1] == "/":  # remove trailing slash
        datadir = datadir[:-1]
    scene_type, scene_name = datadir.split("/")[-2:]
    scratch_root_folder = "/scratch/inf0/user/tretschk/data/"
    scratch_scene_folder = os.path.join(scratch_root_folder, scene_type, scene_name)
    return scratch_scene_folder


def get_scratch_scene_folder_valid_file(datadir):
    import os

    scratch_scene_folder = get_scratch_scene_folder(datadir)
    valid_file = os.path.join(scratch_scene_folder, "VALID_SCRATCH")
    return valid_file


def scratch_scene_folder_is_valid(datadir):
    import os

    valid_file = get_scratch_scene_folder_valid_file(datadir)
    return os.path.exists(valid_file)


def check_scratch_for_dataset_copy(datadir):
    if scratch_scene_folder_is_valid(datadir):
        return get_scratch_scene_folder(datadir)
    else:
        return datadir


def overwrite_settings_for_pref(settings):

    settings.optimization_mode = "all"
    settings.pure_mlp_bending = True
    settings.use_temporal_latent_codes = True
    settings.tracking_mode = "plain"
    settings.weight_background_loss = 0.0
    settings.weight_hard_surface_loss = 0.0
    settings.weight_coarse_smooth_deformations = 0.0
    settings.weight_fine_smooth_deformations = 0.0
    settings.activation_function = "ReLU"
    settings.do_zero_out = False
    settings.coarse_and_fine = False
    settings.always_load_full_dataset = True
    settings.num_iterations = 50000
    settings.reconstruction_loss_type = "L2"

    return settings


def overwrite_settings_for_nrnerf(settings):

    settings.optimization_mode = "all"
    settings.pure_mlp_bending = True
    settings.use_temporal_latent_codes = True
    settings.tracking_mode = "plain"
    settings.smooth_deformations_type = "divergence"
    settings.weight_smooth_deformations = 3.0
    settings.weight_background_loss = 0.0
    settings.weight_hard_surface_loss = 0.0
    settings.weight_coarse_smooth_deformations = 0.0
    settings.weight_fine_smooth_deformations = 0.0
    settings.activation_function = "ReLU"
    settings.do_zero_out = False
    settings.coarse_and_fine = False
    settings.always_load_full_dataset = True
    settings.num_iterations = 0
    settings.reconstruction_loss_type = "L2"

    return settings


def overwrite_settings_for_dnerf(settings):

    settings.optimization_mode = "dnerf"
    settings.pure_mlp_bending = True
    settings.use_temporal_latent_codes = False
    settings.tracking_mode = "temporal"
    settings.smooth_deformations_type = "divergence"  # won't be used anyway
    settings.weight_smooth_deformations = 0.0
    settings.weight_background_loss = 0.0
    settings.weight_hard_surface_loss = 0.0
    settings.weight_coarse_smooth_deformations = 0.0
    settings.weight_fine_smooth_deformations = 0.0
    settings.activation_function = "ReLU"
    settings.do_zero_out = False
    settings.coarse_and_fine = False
    settings.fix_coarse_after_a_while = False
    settings.always_load_full_dataset = True
    settings.num_iterations = 800000
    settings.reconstruction_loss_type = "L2"
    settings.use_viewdirs = True

    return settings


class Returns:
    def __init__(self, restricted=None):

        self.mode_dict = {}
        self.mode = None
        self.restricted = restricted  # None takes everything. restricted only what's in restricted.

        self.mask = None

    # more internal stuff

    def get_restricted_list(self):
        return self.restricted

    def activate_mode(self, mode):
        if mode not in self.mode_dict:
            self.mode_dict[mode] = {}
        self.mode = mode

    def get_modes(self):
        return list(self.mode_dict.keys())

    # add, delete

    def set_mask(self, mask):
        self.mask = mask

    def get_mask(self):
        return self.mask

    def add_return(self, name, returns, clone=True, infill=0):
        if self.restricted is None or name in self.restricted:
            if clone and self.mask is None:
                returns = returns.clone()
            if self.mask is not None:
                if not clone:
                    raise RuntimeError
                returns = infill_masked(self.mask, returns, infill_value=infill)
            self.mode_dict[self.mode][name] = returns
            successful = True
            return successful
        else:
            successful = False
            return successful

    def add_returns(self, returns_dict):
        for name, returns in returns_dict.items():
            self.add_return(name, returns)

    def delete_return(self, name):
        del self.mode_dict[self.mode][name]

    # contains, get

    def __contains__(self, name):
        return name in self.mode_dict[self.mode]

    def get_returns(self, mode=None):
        if mode is None:
            mode = self.mode
        return self.mode_dict[mode]

    # modify

    def concatenate_returns(self, modes=None):
        if modes is None:
            modes = self.get_modes()

        returns = {
            key: torch.cat([self.mode_dict[mode][key] for mode in modes], axis=0)
            for key in self.mode_dict[modes[0]].keys()
        }
        return returns

    def pull_to_cpu(self):
        self.mode_dict[self.mode] = {
            key: tensor.cpu() for key, tensor in self.mode_dict[self.mode].items()
        }

    def push_to_gpu(self):
        self.mode_dict[self.mode] = {
            key: tensor.cuda() for key, tensor in self.mode_dict[self.mode].items()
        }

    def reshape_returns(self, height, width):
        self.mode_dict[self.mode] = {
            key: tensor.view(size=(height, width) + tensor.shape[1:])
            for key, tensor in self.mode_dict[self.mode].items()
        }
