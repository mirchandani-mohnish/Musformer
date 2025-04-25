#!/usr/bin/env/python3
"""
Recipe for vocal separation using convtasnet
"""

import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
import pdb

import musdb
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import speechbrain as sb
import psutil
from dataset import MusDBDataset


import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.core import AMPConfig
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
import time
from torch.utils.data import DataLoader

import musdb


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix       
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)      

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_sources)],
            dim=-1,
        ).to(self.device)

        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_wavedrop:
                    mix = self.hparams.drop_chunk(mix, mix_lens)
                    mix = self.hparams.drop_freq(mix)
        
        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_sources)
        sep_h = mix_w * est_mask
        
        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_sources)
            ],
            dim=-1,
        )

        # pad estimates as per requirement 
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        # print("INSIDE FIT BATCH")
        
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0
        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.voc_sig, batch.inst_sig] #mix_sig, voc_sig, inst_sig
       
        
        predictions, targets = self.compute_forward(
            mixture, targets, sb.Stage.TRAIN
        )
        loss = self.compute_objectives(predictions, targets)

        if self.hparams.threshold_byloss:
            th = self.hparams.threshold
            loss = loss[loss > th]
            if loss.nelement() > 0:
                loss = loss.mean()
        else:
            loss = loss.mean()

        if (
            loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
        ):  # the fix for computational problems
            loss.backward()
            if self.hparams.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(),
                    self.hparams.clip_grad_norm,
                )
            self.optimizer.step()
        else:
            self.nonfinite_count += 1
            logger.info(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0.0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.track_id
        mixture = batch.mix_sig
        targets = [batch.voc_sig, batch.inst_sig]

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_loader):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]


        def is_silent(source, threshold=1e-6):
            return np.max(np.abs(source[0])) < threshold or np.max(np.abs(source[1])) < threshold

        with open(save_file, "w", newline="", encoding="utf-8") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()
            skip_cnt = 0

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.track_id
                    targets = [batch.voc_sig, batch.inst_sig]
                   

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_sources, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline
                    
     
                    if not is_silent(targets[0].t().cpu().numpy()) and not is_silent(predictions[0].t().detach().cpu().numpy()) and not is_silent(mixture_signal[0].t().detach().cpu().numpy()):
                        
                    
                        sdr, _, _, _ = bss_eval_sources(
                            targets[0].t().cpu().numpy(),
                            predictions[0].t().detach().cpu().numpy(),
                            compute_permutation=False
                        )
    
                        sdr_baseline, _, _, _ = bss_eval_sources(
                            targets[0].t().cpu().numpy(),
                            mixture_signal[0].t().detach().cpu().numpy(),
                            compute_permutation=False
                        )
    
                        sdr_i = sdr.mean() - sdr_baseline.mean()
    
                        # Saving on a csv file
                        row = {
                            "snt_id": snt_id[0],
                            "sdr": sdr.mean(),
                            "sdr_i": sdr_i,
                            "si-snr": -sisnr.item(),
                            "si-snr_i": -sisnr_i.item(),
                        }
                        writer.writerow(row)
    
                        # Metric Accumulation
                        all_sdrs.append(sdr.mean())
                        all_sdrs_i.append(sdr_i.mean())
                        all_sisnrs.append(-sisnr.item())
                        all_sisnrs_i.append(-sisnr_i.item())
    
                    
                else:
                    skip_cnt += 1
                    print(f"Warning: skipping silent target, this has happened {skip_cnt} times")
                
                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))
        if(self.hparams.result_file_path is not None and self.hparams.result_file_path != ""):
            with open(self.hparams.result_file_path, "a", newline="", encoding="utf-8") as metrics_csv:
                writer = csv.DictWriter(metrics_csv, fieldnames=["model_name", "n_epochs", "learning_rate", "chunk_size", "sample_rate", "sdr", "sdr_i", "si-snr", "si-snr_i"])
                row = {
                        "model_name": self.hparams.experiment_name,
                        "learning_rate": self.hparams.lr,
                        "n_epochs": self.hparams.N_epochs,
                        "chunk_size":self.hparams.chunk_size,
                        "sample_rate":self.hparams.sample_rate,
                        "sdr": np.array(all_sdrs).mean(),
                        "sdr_i": np.array(all_sdrs_i).mean(),
                        "si-snr": np.array(all_sisnrs).mean(),
                        "si-snr_i": np.array(all_sisnrs_i).mean(),
                    }
                writer.writerow(row)

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create output folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_sources):
            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )





if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = get_logger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Update precision to bf16 if the device is CPU and precision is fp16
    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

   

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
 
    # Training
        
    # Usage with SpeechBrain
    train_data = MusDBDataset(hparams["db_path"], subset="train", split="train", target_sr=hparams["sample_rate"], chunk_size=hparams["chunk_size"])
    valid_data = MusDBDataset(hparams["db_path"], subset="train", split="valid", target_sr=hparams["sample_rate"], chunk_size=hparams["chunk_size"])
    test_data = MusDBDataset(hparams["db_path"], subset="test", target_sr=hparams["sample_rate"], chunk_size=hparams["chunk_size"])
    

    # Create DataLoader
    train_loader = sb.dataio.dataloader.make_dataloader(
        train_data,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=sb.dataio.batch.PaddedBatch  # Handles variable lengths
    )

    valid_loader = sb.dataio.dataloader.make_dataloader(
        valid_data,
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=sb.dataio.batch.PaddedBatch  # Handles variable lengths
    )

    test_loader = sb.dataio.dataloader.make_dataloader(
        test_data,
        batch_size=hparams["batch_size"],
        collate_fn=sb.dataio.batch.PaddedBatch  # Handles variable lengths
    )
    
    separator.fit(
        separator.hparams.epoch_counter,
        train_loader,
        valid_loader,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # # Eval
    separator.evaluate(test_loader, min_key="si-snr")
    separator.save_results(test_loader)
