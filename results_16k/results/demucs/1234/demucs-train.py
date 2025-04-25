#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on the wsjmix
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""
## CHECKPOINT
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm



import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from speechbrain.nnet.CNN import Conv1d, ConvTranspose1d
# from speechbrain.nnet.activations import GLU
from speechbrain.lobes.models.beats import GLU_Linear
from torch.nn import GLU
from speechbrain.nnet.RNN import LSTM
from speechbrain.nnet.linear import Linear
from demucsModels import EncoderBlock, DecoderBlock
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper
from dataset import MusDBDataset

from torch.utils.data import Dataset
import musdb
np.float_ = np.float64





# Define training procedure
class DemucsSeparation(sb.Brain):
   


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
        
        mix=mix.unsqueeze(1)
        targets=targets.permute(0,2,1)
       

        mix_enc_1 = self.modules.encoder1(mix)
        mix_enc_2 = self.modules.encoder2(mix_enc_1)
        mix_enc_3 = self.modules.encoder3(mix_enc_2)
        mix_enc_4 = self.modules.encoder4(mix_enc_3)
        mix_enc_5 = self.modules.encoder5(mix_enc_4)
        mix_enc_6 = self.modules.encoder6(mix_enc_5)

        lstm_in = mix_enc_6.permute(0,2,1)
        lstm_out, _ = self.modules.lstm(lstm_in) # outputs both -- outputs as well as hidden states -- we dont need hidden states
        lin_out = self.modules.linear(lstm_out)
        lin_out = lin_out.permute(0,2,1)

        mix_dec_6 = self.modules.decoder6(lin_out, skip=mix_enc_6)
        mix_dec_5 = self.modules.decoder5(mix_dec_6, skip=mix_enc_5)
        mix_dec_4 = self.modules.decoder4(mix_dec_5, skip=mix_enc_4)
        mix_dec_3 = self.modules.decoder3(mix_dec_4, skip=mix_enc_3)
        mix_dec_2 = self.modules.decoder2(mix_dec_3, skip=mix_enc_2)
        mix_dec_1 = self.modules.decoder1(mix_dec_2, skip=mix_enc_1)

        mix_out = self.modules.linearSeparator(mix_dec_1)
        est_source = mix_out



        # T changed after conv1d in encoder, fix it here
        T_origin = targets.size(2)
        T_est = est_source.size(2)

        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, : , :T_origin]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        
        targets = targets.permute(0,2,1)
        predictions = predictions.permute(0,2,1)
        return self.hparams.loss(targets, predictions) # for pitwrapper loss
## CHECKPOINT
    def fit_batch(self, batch):
        """Trains one batch"""

        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.voc_sig, batch.inst_sig]

        with self.training_ctx:
            predictions, targets = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN
            )

            loss = self.compute_objectives(predictions, targets)

            # hard threshold the easy dataitems
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss = loss[loss > th]
                if loss.nelement() > 0:
                    loss = loss.mean()
            else:
                loss = loss.mean()

        # loss threshold and clipping
        if loss.nelement() > 0 and loss < self.hparams.loss_upper_lim:
            self.scaler.scale(loss).backward()
            if self.hparams.clip_grad_norm >= 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(),
                    self.hparams.clip_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
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
                    self.save_audio(snt_id, mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id, mixture, targets, predictions)

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

      

        with open(save_file, "w", newline="", encoding="utf-8") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

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
                    predictions = predictions.permute(0,2,1)
                    targets = targets.permute(0,2,1)
                   
                    # sisnr = self.compute_objectives(predictions, targets)
                    sisnr = get_si_snr_with_pitwrapper(predictions,targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_sources, dim=-1
                    )

                    mixture_signal = mixture_signal.to(targets.device)
                    # sisnr_baseline = self.compute_objectives(
                    #     mixture_signal, targets
                    # )
                    sisnr_baseline = get_si_snr_with_pitwrapper(mixture_signal, targets)
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].mean(dim=1).t().cpu().numpy(),
                        predictions[0].mean(dim=1).t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].mean(dim=1).t().cpu().numpy(),
                        mixture_signal[0].mean(dim=1).t().detach().cpu().numpy(),
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

        # save to metrics file if provided
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
        perm_targets = targets.permute(0,2,1)
        perm_predictions = predictions.permute(0,2,1)
        for ns in range(self.hparams.num_sources):
            
            
            
            signal = perm_predictions[0, :, ns]
           
            signal = signal / signal.abs().max()
           
           
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = perm_targets[0, :, ns]
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


    # Brain class initialization
    separator = DemucsSeparation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    # Training
    separator.fit(
        separator.hparams.epoch_counter,
        train_loader,
        valid_loader,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Eval
    separator.evaluate(test_loader, min_key="si-snr")
    separator.save_results(test_loader)
    ## CHECKPOINT
