#!/usr/bin/env python3
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path
from collections import defaultdict
from omegaconf import OmegaConf
from PIL import Image
from io import BytesIO
import json
from datetime import date
import os
import yaml
from functools import lru_cache

from CDARL.data.shapes3d_data import Shape3dDataset
from CDARL.utils import ExpDataset, reparameterize, RandomTransform, Results, seed_everything
from torchvision.utils import save_image

from experiment_utils import (
    initialize_experiment,
    save_model,
    logger,
    create_optimizer_and_scheduler,
    set_manifold_thickness,
    compute_metrics_on_dataset,
    reset_optimizer_state,
    update_dict,
    optimizer_step,
    step_schedules,
    determine_graph_learning_settings,
    frequency_check,
)
from model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from model import GaussianEncoder, ImageEncoder, ImageDecoder, CoordConv2d, Encoder3dshapes, Decoder3dshapes
from training import VAEMetrics

def create_model_reduce_dim(cfg):
    # Create model
    scm = create_img_scm()
    encoder, decoder = create_img_encoder_decoder(cfg)
    intervention_encoder = create_intervention_encoder(cfg)
    model = ILCM(
            scm,
            encoder=encoder,
            decoder=decoder,
            intervention_encoder=intervention_encoder,
            intervention_prior=None,
            averaging_strategy='stochastic',
            dim_z=16,
            )
    return model

def create_img_scm():
    scm = MLPImplicitSCM(
            graph_parameterization='none',
            manifold_thickness=0.01,
            hidden_units=100,
            hidden_layers=2,
            homoskedastic=False,
            dim_z=16,
            min_std=0.2,
        )

    return scm

def create_img_encoder_decoder(cfg):
    if cfg.data.training.encoder == 'resnet':
        encoder = ImageEncoder(
                in_resolution=64,
                in_features=3,
                out_features=32,
                hidden_features=32,
                batchnorm=False,
                conv_class=CoordConv2d,
                mlp_layers=2,
                mlp_hidden=128,
                elementwise_hidden=16,
                elementwise_layers=0,
                min_std=1.e-3,
                permutation=0,
                )
        decoder = ImageDecoder(
                in_features=32,
                out_resolution=64,
                out_features=3,
                hidden_features=32,
                batchnorm=False,
                min_std=1.0,
                fix_std=True,
                conv_class=CoordConv2d,
                mlp_layers=2,
                mlp_hidden=128,
                elementwise_hidden=16,
                elementwise_layers=0,
                permutation=0,
                )
    elif cfg.data.training.encoder == 'conv':
        encoder = Encoder3dshapes(
                in_resolution=64,
                in_features=3,
                out_features=16,
                hidden_features=32,
                batchnorm=False,
                conv_class=CoordConv2d,
                mlp_layers=2,
                mlp_hidden=128,
                elementwise_hidden=16,
                elementwise_layers=0,
                min_std=1.e-3,
                permutation=0,
                )
        decoder = Decoder3dshapes(
                in_features=16,
                out_resolution=64,
                out_features=3,
                hidden_features=32,
                batchnorm=False,
                min_std=1.0,
                fix_std=True,
                conv_class=CoordConv2d,
                mlp_layers=2,
                mlp_hidden=128,
                elementwise_hidden=16,
                elementwise_layers=0,
                permutation=0,
                )
        
    return encoder, decoder

@hydra.main(version_base=None, config_path="config", config_name="ilcm")
def main(cfg):
    """High-level experiment function"""
    log_dir = os.path.join(cfg.data.save_path, str(date.today()) + '_ilcm')
    os.makedirs(log_dir, exist_ok=True)

    # save config
    with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
        yaml.dump(OmegaConf.to_yaml(cfg), f)

    # Create logs
    results = Results(title="ILCM loss", xlabel="training_step", ylabel="loss")
    results.create_logs(labels=["training_step", "loss", "train_lr"], init_values=[[], [], []])

    # download reduce dim model
    model_reduce_dim = create_model_reduce_dim(cfg)
    weights = torch.load(cfg.data.encoder_path, map_location=torch.device('cuda'))
    for k in list(weights.keys()):
        if k not in model_reduce_dim.state_dict().keys():
            del weights[k]
    model_reduce_dim.load_state_dict(weights)
    print("Loaded Weights")

    # Train
    model = create_model(cfg)
    train(cfg, model, model_reduce_dim, results, log_dir)
    save_model(log_dir, model)

    # Save results
    results.save_logs(cfg.data.save_path, str(date.today()))
    results.generate_plot(log_dir, log_dir)

    logger.info("Anders nog iets?")

def create_model(cfg):
    """Instantiates a (learnable) VAE model"""

    # Create model
    logger.info(f"Creating {cfg.model.type} model")
    scm = create_scm(cfg)
    encoder, decoder = create_encoder_decoder(cfg)
    intervention_encoder = create_intervention_encoder(cfg)
    model = ILCM(
            scm,
            encoder=encoder,
            decoder=decoder,
            intervention_encoder=intervention_encoder,
            intervention_prior=None,
            averaging_strategy=cfg.model.averaging_strategy,
            dim_z=10,
            )

    return model

def create_scm(cfg):
    """Creates an SCM"""
    logger.info(f"Creating {cfg.model.scm.type} SCM")

    logger.info(f"Graph parameterization for noise-centric learning: {cfg.model.scm.adjacency_matrix}")
    scm = MLPImplicitSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=10,
            min_std=cfg.model.scm.min_std,
        )

    return scm

def create_encoder_decoder(cfg):
    """Create encoder and decoder"""
    logger.info(f"Creating {cfg.model.encoder.type} encoder / decoder")

    encoder_hidden_layers = cfg.model.encoder.hidden_layers
    encoder_hidden = [cfg.model.encoder.hidden_units for _ in range(encoder_hidden_layers)]
    decoder_hidden_layers = cfg.model.decoder.hidden_layers
    decoder_hidden = [cfg.model.decoder.hidden_units for _ in range(decoder_hidden_layers)]

    encoder = GaussianEncoder(
                hidden=encoder_hidden,
                input_features=16,
                output_features=10,
                fix_std=cfg.model.encoder.fix_std,
                init_std=cfg.model.encoder.std,
                min_std=cfg.model.encoder.min_std,
            )
    decoder = GaussianEncoder(
                hidden=decoder_hidden,
                input_features=10,
                output_features=16,
                fix_std=cfg.model.decoder.fix_std,
                init_std=cfg.model.decoder.std,
                min_std=cfg.model.decoder.min_std,
            )

    return encoder, decoder

def create_intervention_encoder(cfg):
    """Creates an intervention encoder"""
    logger.info(f"Creating {cfg.model.intervention_encoder.type} intervention encoder")

    intervention_encoder = HeuristicInterventionEncoder()
    return intervention_encoder

# noinspection PyTypeChecker
def train(cfg, model, model_reduce_dim, results, log_dir):
    """High-level training function"""
    seed_everything(cfg.general.seed)

    logger.info("Starting training")
    logger.info(f"Training on {cfg.training.device}")
    device = torch.device(cfg.training.device)

    # Training
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    optim, scheduler = create_optimizer_and_scheduler(cfg, model, separate_param_groups=True)

    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    best_state = {"state_dict": None, "loss": None, "step": None}

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ExpDataset(cfg.data.data_dir, cfg.data.data_tag, cfg.data.num_splitted, transform)
    loader = get_dataloader(cfg, dataset)
    #train_loader = get_dataloader_test(cfg, "test", batchsize=cfg.training.batchsize, shuffle=True)
    #val_loader = get_dataloader_test(cfg, "val", batchsize=cfg.eval.batchsize, shuffle=False, include_noise_encodings=False)
    steps_per_epoch = len(loader)
    print(len(loader))

    # GPU
    model = model.to(device)
    model_reduce_dim = model_reduce_dim.to(device)

    step = 0
    nan_counter = 0
    epoch_generator = trange(cfg.training.epochs, disable=not cfg.general.verbose)

    for epoch in epoch_generator:

        # Graph sampling settings
        graph_kwargs = determine_graph_learning_settings(cfg, epoch, model)

        model_interventions, pretrain, deterministic_intervention_encoder = epoch_schedules(
            cfg, model, model_reduce_dim, epoch, optim
        )

        for i_split in range(cfg.data.num_splitted):
            for i_batch, imgs in enumerate(loader):
            #for z1, z2 in train_loader:

                if step/1000 % 10 == 0:
                    model_interventions, pretrain, deterministic_intervention_encoder = epoch_schedules(
                    cfg, model, model_reduce_dim, step/1000, optim
                    )


                fractional_epoch = step / steps_per_epoch

                x1, x2, z1, z2 = encode_data(cfg, imgs, model_reduce_dim, device)

                model.train()

                (
                    beta,
                    beta_intervention,
                    consistency_regularization_amount,
                    cyclicity_regularization_amount,
                    edge_regularization_amount,
                    inverse_consistency_regularization_amount,
                    z_regularization_amount,
                    intervention_entropy_regularization_amount,
                    intervention_encoder_offset,
                ) = step_schedules(cfg, model, fractional_epoch)

                z1, z2 = (z1.to(device), z2.to(device))

                        # Model forward pass
                log_prob, model_outputs = model(
                            z1,
                            z2,
                            beta=beta,
                            beta_intervention_target=beta_intervention,
                            pretrain_beta=cfg.training.pretrain_beta,
                            full_likelihood=cfg.training.full_likelihood,
                            likelihood_reduction=cfg.training.likelihood_reduction,
                            pretrain=pretrain,
                            model_interventions=model_interventions,
                            deterministic_intervention_encoder=deterministic_intervention_encoder,
                            intervention_encoder_offset=intervention_encoder_offset,
                            **graph_kwargs,
                        )

                    # Loss and metrics
                loss, metrics = criteria(
                            log_prob,
                            z_regularization_amount=z_regularization_amount,
                            edge_regularization_amount=edge_regularization_amount,
                            cyclicity_regularization_amount=cyclicity_regularization_amount,
                            consistency_regularization_amount=consistency_regularization_amount,
                            inverse_consistency_regularization_amount=inverse_consistency_regularization_amount,
                            intervention_entropy_regularization_amount=intervention_entropy_regularization_amount,
                            **model_outputs,
                        )

                    # Optimizer step
                finite, grad_norm = optimizer_step(cfg, loss, model, model_outputs, optim, z1, z2)
                if not finite:
                    nan_counter += 1

                    # Validation loop
                    #if frequency_check(step, cfg.training.validate_every_n_steps):
                        #validation_loop(cfg, model, model_reduce_dim, criteria, imgs, best_state, val_metrics, step, device)

                    # Log loss and metrics
                step += 1
                    #log_training_step(cfg,beta,epoch_generator,finite,grad_norm,metrics,model,step,train_metrics,nan_counter)

                    # Save model checkpoint
            if frequency_check(step, cfg.data.training.save_model_every_n_steps):
                save_model(log_dir, model, f"model_step_{step}_{epoch}.pt")
                imgs1 = x1
                with torch.no_grad():
                    recon1 = model_reduce_dim.encode_decode(imgs1)
                saved_imgs = torch.cat([imgs1, recon1], dim=0)
                # save images
                path_image = os.path.join(log_dir, f'recon_{step}.png')
                save_image(saved_imgs, path_image, nrow=10)
                results.update_logs(["training_step", "loss", "train_lr"], [step, loss.item(), scheduler.get_last_lr()[0]])
                results.save_logs(log_dir)
                #results.generate_plot(log_dir,log_dir)
                    # save metrics
                update_dict(train_metrics, metrics)
                with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
                    json.dump(train_metrics, f)

                updateloader(cfg, loader, dataset)

        if scheduler is not None and epoch < cfg.data.training.epochs - 1:
            scheduler.step()

            # Optionally reset Adam stats
            if (
                cfg.training.lr_schedule.type == "cosine_restarts_reset"
                and (epoch + 1) % cfg.training.lr_schedule.restart_every_epochs == 0
                and epoch + 1 < cfg.data.training.epochs
            ):
                logger.info(f"Resetting optimizer at epoch {epoch + 1}")
                reset_optimizer_state(optim)

        # noinspection PyTypeChecker
        if cfg.training.early_stopping and best_state["step"] < step:
            logger.info(
                f'Early stopping after step {best_state["step"]} '
                f'with validation loss {best_state["loss"]}'
            )
            model.load_state_dict(best_state["state_dict"])

    # Reset model: back to CPU, reset manifold thickness
    set_manifold_thickness(cfg, model, None)

    return train_metrics


def get_dataloader(cfg, dataset):
    """Load data from disk and return DataLoader instance"""
    logger.debug(f"Loading data {cfg.data.name}")
    loader = DataLoader(dataset, batch_size=cfg.data.training.batchsize, shuffle=True, num_workers=cfg.data.num_workers)
    logger.debug(f"Finished loading data {cfg.data.name}")
    return loader

def updateloader(cfg, loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=cfg.data.training.batchsize, shuffle=True, num_workers=cfg.training.num_workers)
    return loader

def get_dataloader_test(cfg, tag, batchsize=None, shuffle=False, include_noise_encodings=False):
    """Load data from disk and return DataLoader instance"""
    filename = Path(cfg.data.data_dir_ilcm) / f"{tag}_carracing_encoded.pt"
    data = torch.load(filename)
    print(len(data))
    dataset = TensorDataset(*data)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)

    return dataloader

def encode_data(cfg, imgs, model_reduce_dim, device):
    #style
    if cfg.data.training.intervention == 'style':
        if cfg.data.training.number_domains == 10:
            imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)               
            imgs = imgs.reshape(-1, *imgs.shape[2:])
            imgs = imgs.repeat(2, 1, 1, 1)
            imgs = RandomTransform(imgs).apply_transformations(nb_class=2, value=[0, 0.1])
            x1 = imgs[0]
            x2 = imgs[1]
        elif cfg.data.training.number_domains == 2: 
            imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)  
            imgs = imgs[2:3, :, :, :, :]
            imgs = imgs.reshape(-1, *imgs.shape[2:])
            imgs = imgs.repeat(2, 1, 1, 1)
            imgs = RandomTransform(imgs).apply_transformations(nb_class=2, value=[0, -0.2])
            x1 = imgs[0]
            x2 = imgs[1]
        elif cfg.data.training.number_domains == 5:
            imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)  
            imgs = imgs[0:3, :, :, :, :]
            imgs = imgs.reshape(-1, *imgs.shape[2:])
            imgs = imgs.repeat(2, 1, 1, 1)
            imgs = RandomTransform(imgs).apply_transformations(nb_class=2, value=[0, -0.1])
            x1 = imgs[0]
            x2 = imgs[1]

    if cfg.data.training.intervention == 'content':
        if cfg.data.training.number_domains == 10:
            imgs = imgs.to(device, non_blocking=True)
            imgs = imgs.reshape(-1, *imgs.shape[2:])
            imgs = imgs.repeat(2, 1, 1, 1)
            imgs = RandomTransform(imgs).apply_transformations(nb_class=2, value=[0, 0.1])
            imgs = imgs.permute(1,0,2,3,4)
            m = int(imgs.shape[0]/2)
            feature_1 = imgs[:m]
            x1 = feature_1.reshape(-1, *imgs.shape[2:])
            feature_2 = imgs[m:]
            x2 = feature_2.reshape(-1, *imgs.shape[2:])
        elif cfg.data.training.number_domains == 2: 
            imgs = imgs.to(device, non_blocking=True)
            imgs = imgs[2:3, :, :, :, :]
            imgs = imgs.reshape(-1, *imgs.shape[2:])
            imgs = imgs.repeat(2, 1, 1, 1)
            imgs = RandomTransform(imgs).apply_transformations(nb_class=2, value=[0, -0.2])
            imgs = imgs.permute(1,0,2,3,4)
            m = int(imgs.shape[0]/2)
            feature_1 = imgs[:m]
            x1 = feature_1.reshape(-1, *imgs.shape[2:])
            feature_2 = imgs[m:]
            x2 = feature_2.reshape(-1, *imgs.shape[2:])
        elif cfg.data.training.number_domains == 5:
            imgs = imgs.to(device, non_blocking=True)
            m = int(imgs.shape[0]/2)
            feature_1 = imgs[:m]
            x1 = feature_C1.reshape(-1, *imgs.shape[2:])
            feature_2 = imgs[m:]
            x2 = feature_2.reshape(-1, *imgs.shape[2:])

    x1 = imgs[0]
    x2 = imgs[1]
    
    x1, x2 = (x1.to(device), x2.to(device))
    with torch.no_grad():
        _, _, z1, z2, *_ = model_reduce_dim.encode_decode_pair(x1, x2)
    return x1, x2, z1, z2

def epoch_schedules(cfg, model, model_reduce_dim, epoch, optim):
    """Epoch-based schedulers"""

    # Pretraining?
    pretrain = cfg.training.pretrain_epochs is not None and epoch < cfg.training.pretrain_epochs
    if epoch == cfg.training.pretrain_epochs:
        print(f"Stopping pretraining at epoch {epoch}")

    # Model interventions in SCM / noise model?
    model_interventions = (
        cfg.training.model_interventions_after_epoch is None
        or epoch >= cfg.training.model_interventions_after_epoch
    )
    if epoch == cfg.training.model_interventions_after_epoch:
        print(f"Beginning to model intervention distributions at epoch {epoch}")

    # Freeze encoder?
    if cfg.training.freeze_encoder_epoch is not None and epoch == cfg.training.freeze_encoder_epoch:
        logger.info(f"Freezing encoder and decoder at epoch {epoch}")
        optim.param_groups[0]["lr"] = 0.0
        # model.encoder.freeze()
        # model.decoder.freeze()

    # Fix noise-centric model to a topological order encoder?
    if (
        "fix_topological_order_epoch" in cfg.training
        and cfg.training.fix_topological_order_epoch is not None
        and epoch == cfg.training.fix_topological_order_epoch
    ):
        logger.info(f"Determining topological order at epoch {epoch}")
        fix_topological_order(cfg, model, model_reduce_dim, partition="val")

    # Deterministic intervention encoders?
    if cfg.training.deterministic_intervention_encoder_after_epoch is None:
        deterministic_intervention_encoder = False
    else:
        deterministic_intervention_encoder = (
            epoch >= cfg.training.deterministic_intervention_encoder_after_epoch
        )
    if epoch == cfg.training.deterministic_intervention_encoder_after_epoch:
        print(f"Switching to deterministic intervention encoder at epoch {epoch}")

    return model_interventions, pretrain, deterministic_intervention_encoder


@torch.no_grad()
def fix_topological_order(cfg, model, model_reduce_dim, partition="val", dataloader=None):
    """Fixes the topological order in an ILCM"""

    # This is only defined for noise-centric models (ILCMs)
    assert cfg.model.type == "intervention_noise_vae"

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model.to(device)

    # Dataloader
    if dataloader is None:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ExpDataset(cfg.data.data_dir, cfg.data.data_tag, cfg.data.num_splitted, transform)
        loader = get_dataloader(cfg, dataset)
        #dataloader = get_dataloader_test(cfg, partition, cfg.eval.batchsize)

    # Load data and compute noise encodings
    noise = []
    #for x_batch, *_ in dataloader:
    for i_batch, imgs in enumerate(loader):
        _, _, z, _ = encode_data(cfg, imgs, model_reduce_dim, device)
        batch = z.to(z)
        noise.append(model.encode_to_noise(batch, deterministic=True).to(cpu))
        #x_batch = x_batch.to(device)
        #noise.append(model.encode_to_noise(x_batch, deterministic=True).to(cpu))

    noise = torch.cat(noise, dim=0).detach()

    # Median values of each noise component (to be used as dummy values when masking)
    dummy_values = torch.median(noise, dim=0).values
    logger.info(f"Dummy noise encodings: {dummy_values}")

    # Find topological order
    model = model.to(cpu)
    topological_order = find_topological_order(model, noise)
    logger.info(f"Topological order: {topological_order}")

    # Fix topological order
    model.scm.set_causal_structure(
        None, "fixed_order", topological_order=topological_order, mask_values=dummy_values
    )
    model.to(device)

def find_topological_order(model, noise):
    """
    Extracts the topological order from a noise-centric model by iteratively looking for the
    least-dependant solution function
    """

    @lru_cache()
    def solution_dependance_on_noise(i, j):
        """Tests how strongly solution s_i depends on noise variable e_j"""

        transform = model.scm.solution_functions[i]
        inputs = noise[:, i].unsqueeze(1)

        mask_ = torch.ones_like(noise)
        mask_[:, i] = 0
        context = mask(noise, mask_)

        # Note that we need to invert here b/c the transform is defined from z to e
        return dependance(transform, inputs, context, j, invert=True)

    topological_order = []
    components = set(range(model.dim_z))

    while components:
        least_dependant_solution = None
        least_dependant_score = float("inf")

        # For each variable, check how strongly its solution function depends on the other noise
        # vars
        for i in components:
            others = [j for j in components if j != i]
            score = sum(solution_dependance_on_noise(i, j) for j in others)

            if score < least_dependant_score:
                least_dependant_solution = i
                least_dependant_score = score

        # The "least dependant" variable will the be next in our topological order, then we remove
        # it and consider only the remaining vars
        topological_order.append(least_dependant_solution)
        components.remove(least_dependant_solution)

    return topological_order


def dependance(
    transform,
    inputs,
    context,
    component,
    invert=False,
    measure=torch.nn.functional.mse_loss,
    normalize=True,
    **kwargs,
):
    """
    Computes a measure of functional dependence of a transform on a given component of the context
    """

    # Shuffle the component of the context
    context_shuffled = context.clone()
    batchsize = context.shape[0]
    idx = torch.randperm(batchsize)
    context_shuffled[:, component] = context_shuffled[idx, component]

    # Compute function with and without permutation
    function = transform.inverse if invert else transform
    f, _ = function(inputs, context=context, **kwargs)
    f_shuffled, _ = function(inputs, context=context_shuffled, **kwargs)

    # Normalize so that this becomes comparable
    if normalize:
        mean, std = torch.mean(f), torch.std(f)
        std = torch.clamp(std, 0.1)
        f = (f - mean) / std
        f_shuffled = (f_shuffled - mean) / std

    # Compute difference
    difference = measure(f, f_shuffled)

    return difference

def mask(data, mask_, mask_data=None, concat_mask=True):
    """Masking on a tensor, optionally adding the mask to the data"""

    if mask_data is None:
        masked_data = mask_ * data
    else:
        masked_data = mask_ * data + (1 - mask_) * mask_data

    if concat_mask:
        masked_data = torch.cat((masked_data, mask_), dim=1)

    return masked_data

def solution_dependance_on_noise(model, i, j, noise):
    """Tests whether solution s_i depends on noise variable e_j"""

    transform = model.scm.solution_functions[i]
    inputs = noise[:, i].unsqueeze(1)

    mask_ = torch.ones_like(noise)
    mask_[:, i] = 0
    context = mask(noise, mask_)

    # Note that we need to invert here b/c the transform is defined from z to e
    return dependance(transform, inputs, context, j, invert=True)

@torch.no_grad()
def validation_loop_imgs(cfg, model, model_reduce_dim, criteria, imgs, best_state, val_metrics, step, device):
    """Validation loop, computing a number of metrics and checkpointing the best model"""

    x1, x2, z1, z2 = encode_data(cfg, imgs, model_reduce_dim, device)

    return eval_implicit_graph(cfg, model, model_reduce_dim, dataloader=imgs)

@torch.no_grad()
def validation_loop(cfg, model, criteria, val_loader, best_state, val_metrics, step, device):
    """Validation loop, computing a number of metrics and checkpointing the best model"""

    loss, nll, metrics = compute_metrics_on_dataset(cfg, model, criteria, val_loader, device)
    metrics.update(eval_implicit_graph(cfg, model, dataloader=val_loader))

    # Store validation metrics
    update_dict(val_metrics, metrics)

    # Print DCI disentanglement score
    print(
        f"Step {step}: causal disentanglement = {metrics['causal_disentanglement']:.2f}, "
        f"noise disentanglement = {metrics['noise_disentanglement']:.2f}"
    )

    # Early stopping: compare val loss to last val loss
    new_val_loss = metrics["nll"] if cfg.training.early_stopping_var == "nll" else loss.item()
    if best_state["loss"] is None or new_val_loss < best_state["loss"]:
        best_state["loss"] = new_val_loss
        best_state["state_dict"] = model.state_dict().copy()
        best_state["step"] = step

@torch.no_grad()
def eval_implicit_graph(cfg, model, model_reduce_dim, partition="val", dataloader=None):
    """Evaluates implicit graph"""

    # This is only defined for noise-centric models (ILCMs)
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}

    # Let's skip this for large latent spaces
    if cfg.model.dim_z > 7:
        return {}

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")

    # Dataloader
    if dataloader is None:
        dataloader = get_dataloader(cfg)

    # Load data and compute noise encodings
    noise = []
    _, _, z1, z2 = encode_data(cfg, dataloader, model_reduce_dim, device)
    x_batch = z1.to(device)
    noise.append(model.encode_to_noise(x_batch, deterministic=True).to(cpu))

    noise = torch.cat(noise, dim=0).detach()

    # Evaluate causal strength
    model = model.to(cpu)
    causal_effects, topological_order = compute_implicit_causal_effects(model, noise)

    # Package as dict
    results = {
        f"implicit_graph_{i}_{j}": causal_effects[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }
    print('implicit graph results: ', results)

    model.to(device)

    return results


def eval_enco_graph(cfg, model, model_reduce_dim, partition="train"):
    """Post-hoc graph evaluation with ENCO"""

    # Only want to do this for ILCMs
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}

    # Let's skip this for large latent spaces
    if cfg.model.dim_z > 7:
        return {}

    logger.info("Evaluating learned graph with ENCO")

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model.to(device)

    # Load data and compute causal variables
    dataloader = get_dataloader(cfg)
    z0s, z1s, interventions = [], [], []

    with torch.no_grad():
        _, _, x0, x1 = encode_data(cfg, dataloader, model_reduce_dim, device)
        x0, x1 = x0.to(device), x1.to(device)
        _, _, _, _, e0, e1, _, _, intervention = model.encode_decode_pair(
                x0.to(device), x1.to(device)
            )
        z0 = model.scm.noise_to_causal(e0)
        z1 = model.scm.noise_to_causal(e1)

        z0s.append(z0.to(cpu))
        z1s.append(z1.to(cpu))
        interventions.append(intervention.to(cpu))

        z0s = torch.cat(z0s, dim=0).detach()
        z1s = torch.cat(z1s, dim=0).detach()
        interventions = torch.cat(interventions, dim=0).detach()

    # Run ENCO
    adjacency_matrix = (
        run_enco(z0s, z1s, interventions, lambda_sparse=cfg.eval.enco_lambda, device=device)
        .cpu()
        .detach()
    )

    # Package as dict
    results = {
        f"enco_graph_{i}_{j}": adjacency_matrix[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }
    print('Enco results: ', results)

    return results


if __name__ == "__main__":
    main()