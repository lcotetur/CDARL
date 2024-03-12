#!/usr/bin/env python3
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from pathlib import Path
from collections import defaultdict
from omegaconf import OmegaConf
from PIL import Image
from io import BytesIO
from datetime import date
import yaml
import json
import os

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
from model import ImageEncoderCarla, ImageDecoderCarla, BasicEncoderCarla, BasicDecoderCarla, CoordConv2d
from training import VAEMetrics

@hydra.main(version_base=None, config_path="config", config_name="reduce_dim_ilcm")
def main(cfg):
    """High-level experiment function"""
    log_dir = os.path.join(cfg.data.save_path, str(date.today()) + f'_{cfg.general.seed}')
    os.makedirs(log_dir, exist_ok=True)

    # save config
    with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
        yaml.dump(OmegaConf.to_yaml(cfg), f)

    # Create logs
    results = Results(title="ILCM loss", xlabel="training_step", ylabel="loss")
    results.create_logs(labels=["training_step", "loss", "train_lr"], init_values=[[], [], []])

    # Train
    model = create_model(cfg)
    train(cfg, model, results, log_dir)
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
    if cfg.data.training.encoder == 'resnet':
        encoder = ImageEncoderCarla(
                in_resolution= 128, #cfg.model.dim_x[2],
                in_features=3, #cfg.model.dim_x[0],
                out_features=cfg.model.dim_z,
                hidden_features=cfg.model.encoder.hidden_channels,
                batchnorm=cfg.model.encoder.batchnorm,
                conv_class=CoordConv2d if cfg.model.encoder.coordinate_embeddings else torch.nn.Conv2d,
                mlp_layers=cfg.model.encoder.extra_mlp_layers,
                mlp_hidden=cfg.model.encoder.extra_mlp_hidden_units,
                elementwise_hidden=cfg.model.encoder.elementwise_hidden_units,
                elementwise_layers=cfg.model.encoder.elementwise_layers,
                min_std=cfg.model.encoder.min_std,
                permutation=cfg.model.encoder.permutation,
                )
        decoder = ImageDecoderCarla(
                in_features=cfg.model.dim_z,
                out_resolution=128, #cfg.model.dim_x[2],
                out_features=3, #cfg.model.dim_x[0],
                hidden_features=cfg.model.decoder.hidden_channels,
                batchnorm=cfg.model.decoder.batchnorm,
                min_std=cfg.model.decoder.min_std,
                fix_std=cfg.model.decoder.fix_std,
                conv_class=CoordConv2d if cfg.model.decoder.coordinate_embeddings else torch.nn.Conv2d,
                mlp_layers=cfg.model.decoder.extra_mlp_layers,
                mlp_hidden=cfg.model.decoder.extra_mlp_hidden_units,
                elementwise_hidden=cfg.model.decoder.elementwise_hidden_units,
                elementwise_layers=cfg.model.decoder.elementwise_layers,
                permutation=cfg.model.encoder.permutation,
                )
    elif cfg.data.training.encoder == 'conv':
        encoder = BasicEncoderCarla(
                in_resolution= 128, #cfg.model.dim_x[2],
                in_features=3, #cfg.model.dim_x[0],
                out_features=cfg.model.dim_z,
                hidden_features=cfg.model.encoder.hidden_channels,
                batchnorm=cfg.model.encoder.batchnorm,
                conv_class=CoordConv2d if cfg.model.encoder.coordinate_embeddings else torch.nn.Conv2d,
                mlp_layers=cfg.model.encoder.extra_mlp_layers,
                mlp_hidden=cfg.model.encoder.extra_mlp_hidden_units,
                elementwise_hidden=cfg.model.encoder.elementwise_hidden_units,
                elementwise_layers=cfg.model.encoder.elementwise_layers,
                min_std=cfg.model.encoder.min_std,
                permutation=cfg.model.encoder.permutation,
                )
        decoder = BasicDecoderCarla(
                in_features=cfg.model.dim_z,
                out_resolution=128, #cfg.model.dim_x[2],
                out_features=3, #cfg.model.dim_x[0],
                hidden_features=cfg.model.decoder.hidden_channels,
                batchnorm=cfg.model.decoder.batchnorm,
                min_std=cfg.model.decoder.min_std,
                fix_std=cfg.model.decoder.fix_std,
                conv_class=CoordConv2d if cfg.model.decoder.coordinate_embeddings else torch.nn.Conv2d,
                mlp_layers=cfg.model.decoder.extra_mlp_layers,
                mlp_hidden=cfg.model.decoder.extra_mlp_hidden_units,
                elementwise_hidden=cfg.model.decoder.elementwise_hidden_units,
                elementwise_layers=cfg.model.decoder.elementwise_layers,
                permutation=cfg.model.encoder.permutation,
                )

    if encoder.permutation is not None:
        logger.info(f"Encoder permutation: {encoder.permutation.detach().numpy()}")
    if decoder.permutation is not None:
        logger.info(f"Decoder permutation: {decoder.permutation.detach().numpy()}")

    return encoder, decoder

def create_intervention_encoder(cfg):
    """Creates an intervention encoder"""
    logger.info(f"Creating {cfg.model.intervention_encoder.type} intervention encoder")

    intervention_encoder = HeuristicInterventionEncoder()
    return intervention_encoder

# noinspection PyTypeChecker
def train(cfg, model, results, log_dir):
    """High-level training function"""
    seed_everything(cfg.general.seed)

    logger.info("Starting training")
    logger.info(f"Training on {cfg.training.device}")
    device = torch.device(cfg.training.device)

    # Training
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    optim, scheduler = create_optimizer_and_scheduler(cfg, model, separate_param_groups=True)

    train_metrics = defaultdict(list)
    best_state = {"state_dict": None, "loss": None, "step": None}

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ExpDataset(cfg.data.data_dir, cfg.data.data_tag, cfg.data.num_splitted, transform)
    loader = get_dataloader(cfg, dataset)
    steps_per_epoch = len(loader)

    # GPU
    model = model.to(device)

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

        for i_split in range(cfg.data.num_splitted):
            for i_batch, imgs in enumerate(loader):


                fractional_epoch = step / steps_per_epoch

                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)
                
                # Content
                imgs_content = imgs.permute(1,0,2,3,4).repeat(2, 1, 1, 1, 1)
                m = int(imgs_content.shape[0]/2)
                imgs_content1 = imgs_content[:m, :, :, :, :]
                imgs_content2 = imgs_content[:m, :, :, :, :]
                imgs_content1 = imgs_content1[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].view(imgs_content1.size())
                imgs_content2 = imgs_content2[[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]].view(imgs_content2.size())

                imgs_content1 = imgs_content1.reshape(-1, *imgs_content1.shape[2:])
                imgs_content2 = imgs_content2.reshape(-1, *imgs_content2.shape[2:])

                # Style
                '''
                imgs_style = imgs.repeat(2, 1, 1, 1, 1)
                n = int(imgs_style.shape[0]/2)
                imgs_style1 = imgs_style[:n, :, :, :, :]
                imgs_style2 = imgs_style[n:, :, :, :, :]
                idx = torch.randperm(3)
                imgs_style1 = imgs_style1[idx].view(imgs_style1.size())
                imgs_style1= imgs_style1.reshape(-1, *imgs_style1.shape[2:])
                imgs_style2= imgs_style2.reshape(-1, *imgs_style2.shape[2:])

                x1 = torch.cat((imgs_style1, imgs_content1), dim=0)
                x2 = torch.cat((imgs_style2, imgs_content2), dim=0)
                '''

                x1 = imgs_content1
                x2 = imgs_content2
                save_image(torch.cat((x1, x2), dim=0), os.path.join(log_dir,'features.png'), nrow=10)

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

                x1, x2 = (x1.to(device), x2.to(device))

                # Model forward pass
                log_prob, model_outputs = model(
                    x1,
                    x2,
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
                #log_training_step(cfg,beta,epoch_generator,finite,grad_norm,metrics,model,step,train_metrics,nan_counter)

                # Save model checkpoint
                if frequency_check(step, cfg.training.save_model_every_n_steps):
                    save_model(log_dir, model, f"model_step_{step}.pt")
                    imgs1 = x1
                    with torch.no_grad():
                        recon1 = model.encode_decode(imgs1)
                    saved_imgs = torch.cat([imgs1, recon1], dim=0)
                    # save images
                    path_image = os.path.join(log_dir, f'recon_{step}.png')
                    save_image(saved_imgs, path_image, nrow=10)

                    results.update_logs(["training_step", "loss", "train_lr"], [step, loss.item(), scheduler.get_last_lr()[0]])
                    results.save_logs(log_dir)
                    # save metrics
                    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
                        json.dump(metrics, f)

            updateloader(cfg, loader, dataset)

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


def get_dataloader(cfg, dataset):
    """Load data from disk and return DataLoader instance"""
    logger.debug(f"Loading data {cfg.data.name}")
    loader = DataLoader(dataset, batch_size=cfg.data.training.batchsize, shuffle=cfg.data.training.shuffle, num_workers=4) #cfg.data.training.batchsize
    logger.debug(f"Finished loading data {cfg.data.name}")
    return loader

def updateloader(cfg, loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=cfg.data.training.batchsize, shuffle=cfg.data.training.shuffle, num_workers=4)
    return loader

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