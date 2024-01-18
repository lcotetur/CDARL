#!/usr/bin/env python3
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra
import torch
from torch.utils.data import Dataset, DataLoader
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

from CDARL.data.shapes3d_data import Shape3dDataset
from CDARL.utils import ExpDataset, reparameterize, RandomTransform, Results, seed_everything, seed_torch
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
from model import GaussianEncoder, ImageEncoder, ImageDecoder, CoordConv2d
from training import VAEMetrics

def create_model_reduce_dim(cfg):
    # Create model
    scm = create_img_scm()
    encoder, decoder = create_img_encoder_decoder()
    intervention_encoder = create_intervention_encoder(cfg)
    model = ILCM(
            scm,
            encoder=encoder,
            decoder=decoder,
            intervention_encoder=intervention_encoder,
            intervention_prior=None,
            averaging_strategy='stochastic',
            dim_z=8,
            )
    return model

def create_img_scm():
    scm = MLPImplicitSCM(
            graph_parameterization='none',
            manifold_thickness=0.01,
            hidden_units=100,
            hidden_layers=2,
            homoskedastic=False,
            dim_z=8,
            min_std=0.2,
        )

    return scm

def create_img_encoder_decoder():
    encoder = ImageEncoder(
            in_resolution=63,
            in_features=3,
            out_features=8,
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
            in_features=8,
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
    log_dir = os.path.join(cfg.data.save_path, str(date.today()))
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
            dim_z=cfg.model.dim_z,
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
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
        )

    return scm

def create_encoder_decoder(cfg):
    """Create encoder and decoder"""
    logger.info(f"Creating {cfg.model.encoder.type} encoder / decoder")

    if cfg.model.encoder.type == "mlp":
        encoder_hidden_layers = cfg.model.encoder.hidden_layers
        encoder_hidden = [cfg.model.encoder.hidden_units for _ in range(encoder_hidden_layers)]
        decoder_hidden_layers = cfg.model.decoder.hidden_layers
        decoder_hidden = [cfg.model.decoder.hidden_units for _ in range(decoder_hidden_layers)]

        encoder = GaussianEncoder(
                hidden=encoder_hidden,
                input_features=cfg.model.dim_x,
                output_features=cfg.model.dim_z,
                fix_std=cfg.model.encoder.fix_std,
                init_std=cfg.model.encoder.std,
                min_std=cfg.model.encoder.min_std,
            )
        decoder = GaussianEncoder(
                hidden=decoder_hidden,
                input_features=cfg.model.dim_z,
                output_features=cfg.model.dim_x,
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
    seed_torch(cfg.general.seed)

    logger.info("Starting training")
    logger.info(f"Training on {cfg.training.device}")
    device = torch.device(cfg.training.device)

    # Training
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    optim, scheduler = create_optimizer_and_scheduler(cfg, model, separate_param_groups=True)

    train_metrics = defaultdict(list)
    best_state = {"state_dict": None, "loss": None, "step": None}

    data = get_dataloader(cfg, batchsize=cfg.data.training.batchsize, shuffle=True)
    steps_per_epoch = 1

    # GPU
    model = model.to(device)
    model_reduce_dim = model_reduce_dim.to(device)

    step = 0
    nan_counter = 0
    epoch_generator = trange(cfg.data.training.epochs, disable=not cfg.general.verbose)
    for epoch in epoch_generator:
        # Graph sampling settings
        graph_kwargs = determine_graph_learning_settings(cfg, epoch, model)

        # Epoch-based schedules
        model_interventions, pretrain, deterministic_intervention_encoder = epoch_schedules(
            cfg, model, epoch, optim
        )

        fractional_epoch = step / steps_per_epoch

        x1, x2, z1, z2 = encode_data(cfg, data, model_reduce_dim, device)

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
        finite, grad_norm = optimizer_step(cfg, loss, model, model_outputs, optim, x1, x2)
        if not finite:
            nan_counter += 1

        # Log loss and metrics
        step += 1
        results.update_logs(["training_step", "loss", "train_lr"], [step, loss.item(), scheduler.get_last_lr()[0]])
        results.save_logs(cfg.data.save_path, str(date.today()))
        #log_training_step(cfg,beta,epoch_generator,finite,grad_norm,metrics,model,step,train_metrics,nan_counter)

        # Save model checkpoint
        if frequency_check(step, cfg.training.save_model_every_n_steps):
            save_model(log_dir, model, f"model_step_{step}.pt")
            imgs1 = x1
            with torch.no_grad():
                recon1 = model_reduce_dim.encode_decode(imgs1)
            saved_imgs = torch.cat([imgs1, recon1], dim=0)
            # save images
            path_image = os.path.join(log_dir, f'recon_{step}.png')
            save_image(saved_imgs, path_image, nrow=10)
            results.generate_plot(log_dir,log_dir)
            # save metrics
            with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f)

            # LR scheduler
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

    # Reset model: back to CPU, reset manifold thickness
    set_manifold_thickness(cfg, model, None)

    return train_metrics


def get_dataloader(cfg, batchsize=32, shuffle=False, include_noise_encodings=False):
    """Load data from disk and return DataLoader instance"""
    logger.debug(f"Loading data {cfg.data.name}")
    dataset = Shape3dDataset()
    dataset.load_dataset(file_dir=cfg.data.data_dir)
    logger.debug(f"Finished loading data {cfg.data.name}")
    return dataset

def encode_data(cfg, data, model_reduce_dim, device):
    imgs = data.create_weak_vae_batch(cfg.data.training.batchsize, cfg.training.device)
    m = int(imgs.shape[2]/2) # 64
    x1 = imgs[:, :, :m, :] # torch.Size([10, 3, 64, 64])
    x2 = imgs[:, :, m:, :] # torch.Size([10, 3, 64, 64])
    x1, x2 = (x1.to(device), x2.to(device))
    with torch.no_grad():
        _, _, z1, z2, *_ = model_reduce_dim.encode_decode_pair(x1, x2)
    return x1, x2, z1, z2

def epoch_schedules(cfg, model, epoch, optim):
    """Epoch-based schedulers"""
    # Pretraining?
    pretrain = cfg.training.pretrain_epochs is not None and epoch < cfg.training.pretrain_epochs
    if epoch == cfg.training.pretrain_epochs:
        logger.info(f"Stopping pretraining at epoch {epoch}")

    # Model interventions in SCM / noise model?
    model_interventions = (
        cfg.training.model_interventions_after_epoch is None
        or epoch >= cfg.training.model_interventions_after_epoch
    )
    if epoch == cfg.training.model_interventions_after_epoch:
        logger.info(f"Beginning to model intervention distributions at epoch {epoch}")

    # Freeze encoder?
    if cfg.training.freeze_encoder_epoch is not None and epoch == cfg.training.freeze_encoder_epoch:
        logger.info(f"Freezing encoder and decoder at epoch {epoch}")
        optim.param_groups[0]["lr"] = 0.0
        # model.encoder.freeze()
        # model.decoder.freeze()

    # Deterministic intervention encoders?
    if cfg.training.deterministic_intervention_encoder_after_epoch is None:
        deterministic_intervention_encoder = False
    else:
        deterministic_intervention_encoder = (epoch >= cfg.training.deterministic_intervention_encoder_after_epoch)
    if epoch == cfg.training.deterministic_intervention_encoder_after_epoch:
        logger.info(f"Switching to deterministic intervention encoder at epoch {epoch}")

    return model_interventions, pretrain, deterministic_intervention_encoder


if __name__ == "__main__":
    main()