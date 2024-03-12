#!/usr/bin/env python3
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All rights reserved.

import hydra
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
from training import VAEMetrics, compute_dci
from enco import run_enco

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
            dim_z=32,
            )
    return model

def create_img_scm():
    scm = MLPImplicitSCM(
            graph_parameterization='none',
            manifold_thickness=0.01,
            hidden_units=100,
            hidden_layers=2,
            homoskedastic=False,
            dim_z=32,
            min_std=0.2,
        )

    return scm

def create_img_encoder_decoder():
    encoder = Encoder3dshapes(
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
    decoder = Decoder3dshapes(
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
    return encoder, decoder

@hydra.main(version_base=None, config_path="config", config_name="ilcm")
@torch.no_grad()
def save_representations(cfg):
    """Reduces dimensionality for full dataset by pushing images through encoder"""
    print("Encoding full datasets and storing representations")
    data = get_dataloader(cfg)

    model_reduce_dim = create_model_reduce_dim(cfg)
    weights = torch.load(cfg.data.encoder_path, map_location=torch.device('cuda'))
    for k in list(weights.keys()):
        if k not in model_reduce_dim.state_dict().keys():
            del weights[k]
    model_reduce_dim.load_state_dict(weights)
    print("Loaded Weights")

    device = torch.device(cfg.training.device)
    model_reduce_dim.to(device)

    for partition in ["train", "test", "val"]:
        z0s, z1s, true_z0s, true_z1s = [], [], [], []
        intervention_labels, interventions, true_e0s, true_e1s = [], [], [], []

        for epoch in range(1000):
            print(epoch)
            if partition == "train":
                imgs, true_latents = data.create_weak_vae_batch_with_true_latents(50, cfg.training.device)
            elif partion == "test" or partition == "val":
                imgs, true_latents = data.create_weak_vae_batch_with_true_latents(5, cfg.training.device)
            m = int(imgs.shape[2]/2) # 64
            x0 = imgs[:, :, :m, :]
            x1 = imgs[:, :, m:, :]
            true_z0 = true_latents[:, :6]
            true_z1 = true_latents[:, 6:]

            x0, x1 = x0.to(device), x1.to(device)
            true_z0, true_z1 = true_z0.to(device), true_z1.to(device)

            _, _, z0, z1, *_ = model_reduce_dim.encode_decode_pair(x0, x1)

            z0s.append(z0)
            z1s.append(z1)
            true_z0s.append(true_z0)
            true_z1s.append(true_z1)

        z0s = torch.cat(z0s, dim=0)
        z1s = torch.cat(z1s, dim=0)
        true_z0s = torch.cat(true_z0s, dim=0)
        true_z1s = torch.cat(true_z1s, dim=0)

        data = (
            z0s,
            z1s,
            true_z0s,
            true_z1s
        )

        filename = Path(cfg.general.exp_dir).resolve() / f"data/{partition}_encoded.pt"
        print(f"Storing encoded {partition} data at {filename}")
        torch.save(data, filename)


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
    results.create_logs(labels=["training_step", "loss", "train_lr", "dci_score", "graph_metric"], init_values=[[], [], [], [], []])

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
    metrics = evaluate(cfg, model, model_reduce_dim, log_dir)

    logger.info("Anders nog iets?")
    return metrics

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

    data = get_dataloader(cfg)
    val_loader = get_dataloader(cfg)
    steps_per_epoch = 1

    # GPU
    model = model.to(device)
    model_reduce_dim = model_reduce_dim.to(device)

    step = 0
    nan_counter = 0
    epoch_generator = trange(cfg.training.epochs, disable=not cfg.general.verbose)
    for epoch in epoch_generator:
        # Graph sampling settings
        graph_kwargs = determine_graph_learning_settings(cfg, epoch, model)

        # Epoch-based schedules
        model_interventions, pretrain, deterministic_intervention_encoder = epoch_schedules(
            cfg, model, model_reduce_dim, epoch, optim
        )

        fractional_epoch = step / steps_per_epoch

        x1, x2, z1, z2, true_z1, true_z2 = encode_data(cfg, data, model_reduce_dim, device)

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

        # Log loss and metrics
        step += 1
        #log_training_step(cfg,beta,epoch_generator,finite,grad_norm,metrics,model,step,train_metrics,nan_counter)

        # Validation loop
        if frequency_check(step, cfg.training.validate_every_n_steps):
            validation_loop(cfg, model, model_reduce_dim, criteria, val_loader, best_state, val_metrics, step, device)

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
            if 'causal_disentanglement' in metrics.keys():
                results.update_logs(["training_step", "loss", "train_lr", "dci_score"], [step, loss.item(), scheduler.get_last_lr()[0], metrics['causal_disentanglement']])
            else:
                results.update_logs(["training_step", "loss", "train_lr", "dci_score"], [step, loss.item(), scheduler.get_last_lr()[0], 0])
            results.save_logs(log_dir)
            results.generate_plot(log_dir,log_dir)
            # save metrics
            update_dict(train_metrics, metrics)
            with open(os.path.join(log_dir, 'train_metrics.json'), 'w') as f:
                json.dump(train_metrics, f)
            with open(os.path.join(log_dir, 'val_metrics.json'), 'w') as f:
                json.dump(val_metrics, f)

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

    # Final validation loop, wrapping up early stopping
    if cfg.training.validate_every_n_steps is not None and cfg.training.validate_every_n_steps > 0:
        validation_loop(cfg, model, model_reduce_dim, criteria, val_loader, best_state, val_metrics, step, device)

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


def get_dataloader(cfg):
    """Load data from disk and return DataLoader instance"""
    logger.debug(f"Loading data {cfg.data.name}")
    dataset = Shape3dDataset()
    dataset.load_dataset(file_dir=cfg.data.data_dir)
    logger.debug(f"Finished loading data {cfg.data.name}")
    return dataset

def encode_data(cfg, data, model_reduce_dim, device):
    #imgs = data.create_weak_vae_batch(cfg.data.training.batchsize, cfg.training.device)
    imgs, true_latents = data.create_weak_vae_batch_with_true_latents(cfg.data.training.batchsize, cfg.training.device)
    m = int(imgs.shape[2]/2) # 64
    x1 = imgs[:, :, :m, :] # torch.Size([10, 3, 64, 64])
    x2 = imgs[:, :, m:, :] # torch.Size([10, 3, 64, 64])
    true_z1 = true_latents[:, :6]
    true_z2 = true_latents[:, 6:]
    with torch.no_grad():
        _, _, z1, z2, *_ = model_reduce_dim.encode_decode_pair(x1, x2)
    return x1, x2, z1, z2, true_z1, true_z2

@torch.no_grad()
def validation_loop(cfg, model, model_reduce_dim, criteria, val_loader, best_state, val_metrics, step, device):
    """Validation loop, computing a number of metrics and checkpointing the best model"""

    x1, x2, z1, z2, true_z1, true_z2 = encode_data(cfg, val_loader, model_reduce_dim, device)

    loss, nll, metrics = compute_metrics_on_dataset(cfg, model, criteria, (z1, z2, true_z1, true_z2), device)
    metrics.update(eval_dci_scores(cfg, model, model_reduce_dim, test_loader=val_loader))
    metrics.update(eval_implicit_graph(cfg, model, model_reduce_dim, dataloader=val_loader))

    update_dict(val_metrics, metrics)
    # Print DCI disentanglement score
    print(f"Causal disentanglement = {metrics['causal_disentanglement']:.2f}, ")

    # Early stopping: compare val loss to last val loss
    new_val_loss = metrics["nll"] if cfg.training.early_stopping_var == "nll" else loss.item()
    if best_state["loss"] is None or new_val_loss < best_state["loss"]:
        best_state["loss"] = new_val_loss
        best_state["state_dict"] = model.state_dict().copy()
        best_state["step"] = step


def evaluate(cfg, model, model_reduce_dim, log_dir):
    """High-level test function"""

    logger.info("Starting evaluation")

    # Compute metrics
    test_metrics = eval_dci_scores(cfg, model, model_reduce_dim, partition=cfg.eval.eval_partition)
    test_metrics.update(eval_enco_graph(cfg, model, model_reduce_dim, partition=cfg.eval.eval_partition))
    test_metrics.update(eval_implicit_graph(cfg, model, model_reduce_dim, partition=cfg.eval.eval_partition))
    test_metrics.update(eval_test_metrics(cfg, model, model_reduce_dim))

    print(f"Final evaluation: causal disentanglement = {test_metrics['causal_disentanglement']:.2f}, ")
    print(f"noise disentanglement = {test_metrics['noise_disentanglement']:.2f}")
    # Store results in csv file
    # Pandas does not like scalar values, have to be iterables
    test_metrics_ = {key: [val] for key, val in test_metrics.items()}
    df = pd.DataFrame.from_dict(test_metrics_)
    df.to_csv(os.path.join(log_dir, "test_metrics.csv"))

    return test_metrics


@torch.no_grad()
def eval_test_metrics(cfg, model, model_reduce_dim):
    """Evaluates loss terms on test data"""

    device = torch.device(cfg.training.device)
    model = model.to(device)

    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    test_loader = get_dataloader(cfg)
    x1, x2, z1, z2, true_z1, true_z2 = encode_data(cfg, data, model_reduce_dim, device)

    _, _, metrics = compute_metrics_on_dataset(
        cfg, model, criteria, (z1, z2, true_z1, true_z2), device=torch.device(cfg.training.device)
    )
    return metrics


@torch.no_grad()
def eval_dci_scores(cfg, model, model_reduce_dim, partition="val", test_loader=None, full_importance_matrix=True):
    """Evaluates DCI scores"""

    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model = model.to(device)

    def _load(partition, device, out_device, dataloader=None):
        if dataloader is None:
            dataloader = get_dataloader(cfg)

        model_z, true_z = [], []

        _, _, x_batch, _, true_z_batch, _ = encode_data(cfg, dataloader, model_reduce_dim, device)

        x_batch = x_batch.to(device)

        z_batch = model.encode_to_causal(x_batch, deterministic=True)

        model_z.append(z_batch.to(out_device))
        true_z.append(true_z_batch.to(out_device))

        model_z = torch.cat(model_z, dim=0).detach()
        true_z = torch.cat(true_z, dim=0).detach()
        return true_z, model_z

    train_true_z, train_model_z = _load("dci_train", device, cpu)
    test_true_z, test_model_z = _load(partition, device, cpu, dataloader=test_loader)

    causal_dci_metrics = compute_dci(
        train_true_z,
        train_model_z,
        test_true_z,
        test_model_z,
        return_full_importance_matrix=full_importance_matrix,
    )

    combined_metrics = {}
    for key, val in causal_dci_metrics.items():
        combined_metrics[f"causal_{key}"] = val

    return combined_metrics


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
    _, _, z1, z2, _, _ = encode_data(cfg, dataloader, model_reduce_dim, device)
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
        for batch in range(100):
            _, _, x0, x1, *_ = encode_data(cfg, dataloader, model_reduce_dim, device)
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

def epoch_schedules(cfg, model, model_reduce_dim, epoch, optim):
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
        logger.info(f"Switching to deterministic intervention encoder at epoch {epoch}")

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

    if dataloader is None:
        data = get_dataloader(cfg)

    # Load data and compute noise encodings
    noise = []
    x1, x2, z1, z2, true_z1, true_z2 = encode_data(cfg, data, model_reduce_dim, device)
    batch = z1.to(device)
    noise.append(model.encode_to_noise(batch, deterministic=True).to(cpu))

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

class CausalMechanism(torch.nn.Module):
    """Causal mechanism extracted from a solution function learned by an ILCM"""

    def __init__(self, solution_transform, component, ancestor_mechanisms):
        super().__init__()

        self.component = component
        self.solution_transform = solution_transform
        self.ancestor_mechanisms = ancestor_mechanisms

    def forward(self, inputs, context, noise, computed_noise=None):
        """Transforms noise (and parent causal variables) to causal variable"""

        solution_context = self._compute_context(inputs, context, noise, computed_noise)

        # Note that the solution transform implements z -> e, here we want forward to mean e -> z
        return self.solution_transform.inverse(inputs, context=solution_context)

    def inverse(self, inputs, context, noise, computed_noise=None):
        """Transforms causal variable (and parent causal variables) to noise"""

        solution_context = self._compute_context(inputs, context, noise, computed_noise)

        # Note that the solution transform implements z -> e, here we want forward to mean e -> z
        return self.solution_transform(inputs, context=solution_context)

    def _compute_context(self, inputs, context, noise, computed_noise=None):
        # Random noise for non-ancestors
        noise = self._randomize_noise(noise)

        # Compute noise encodings corresponding to ancestors
        if computed_noise is None:
            computed_noise = dict()

        for a, mech in self.ancestor_mechanisms.items():
            if a not in computed_noise:
                # print(f'{self.component} -> {a}')
                this_noise, _ = mech.inverse(
                    context[:, a].unsqueeze(1), context, noise, computed_noise=computed_noise
                )
                computed_noise[a] = this_noise.squeeze()

            noise[:, a] = computed_noise[a]

        return noise

    def _randomize_noise(self, noise):
        noise = noise.clone()
        for k in range(noise.shape[1]):
            noise[:, k] = noise[torch.randperm(noise.shape[0]), k]

        return noise


def construct_causal_mechanisms(model, topological_order):
    """Extracts causal mechanisms from model given a topological order"""
    causal_mechanisms = {}

    for i in topological_order:
        solution = model.scm.get_masked_solution_function(i)
        causal_mechanisms[i] = CausalMechanism(
            solution,
            component=i,
            ancestor_mechanisms={a: mech for a, mech in causal_mechanisms.items()},
        )

    return causal_mechanisms


def compute_implicit_causal_effects(model, noise):
    """Tests whether a causal mechanism f_i depends on a particular causal variable z_j"""

    model.eval()

    z = model.scm.noise_to_causal(noise)
    causal_effect = torch.zeros((model.dim_z, model.dim_z))
    # causal_effect[j,i] quantifies how strongly z_j influences z_i

    topological_order = find_topological_order(model, noise)
    mechanisms = construct_causal_mechanisms(model, topological_order)

    for pos, i in enumerate(topological_order):
        for j in topological_order[:pos]:
            causal_effect[j, i] = dependance(
                mechanisms[i], noise[:, i : i + 1], z, j, invert=False, noise=noise
            )

    return causal_effect, topological_order


if __name__ == "__main__":
    #main()
    save_representations()