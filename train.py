import argparse
import logging
import os
from pathlib import Path

import transformers
import diffusers
import torch
import numpy as np
from safetensors.torch import load_file
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.video_processor import VideoProcessor
from diffusers.training_utils import compute_density_for_timestep_sampling


from skyreels_v2_infer.modules.modules import get_vae, get_text_encoder
from skyreels_v2_infer.modules.transformer import WanModel
from skyreels_v2_infer.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler






def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Auto-Regressive Diffusion training script.")


    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument("--seed", type=int, default=2020, help="A seed for reproducible training.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="df_model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs_df",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    parser.add_argument(
        "--allow_tf32",
        # action="store_true",
        type=bool,
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )

    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")

    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )

    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")

    parser.add_argument("--adam_weight_decay", type=float, default=1e-03, help="Weight decay to use.")  # 1e-02

    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )

    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )

    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,  # 1e-5 / 1.0
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument("--num_train_epochs", type=int, default=16)

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )

    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        #default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        default=1000
    )

    args = parser.parse_args()
    return args

def ffop_timestep_scheduler(T, F, t):
    '''
    the ffop_timestep_scheduler is Frame-oriented Probability Propagation (FoPP) timestep scheduler for Diffusion Forcing Training
    https://arxiv.org/pdf/2503.07418
    Args:
        T (int):total timesteps
        F (int):total frames
        t (int):timestep sample from (0, T)
    '''
    #compute the ds and de matrix
    mat_s = np.zeros((T, F))
    mat_e = np.zeros((T, F))
    # set the last frame to 1
    for t in range(T):
        mat_s[t, F - 1] = 1
    # from end to start for f to compute ds
    for f in range(F - 2, -1, -1):
        mat_s[T - 1, f] = 1
        for t in range(T - 2, -1, -1):
            mat_s[t, f] = mat_s[t + 1, f] + mat_s[t, f + 1]
    # set the begin frame to 1
    for t in range(T):
        mat_e[t, 0] = 1
    # from start to end for f to compute de
    for f in range(1, F):
        mat_e[0, f] = 1
        for t in range(1, T):
            mat_e[t, f] = mat_e[t - 1, f] + mat_e[t, f - 1]
    
    # compute the timesteps matrix on f dim
    timesteps_matrix = np.zeros(F)
    curf = np.random.randint(T)
    timesteps_matrix[curf] = t
    
    # get the timesteps from f-1 to 0
    for f in range(curf - 1, -1, -1):
        # print(f, timesteps[f+1], self.mat_e)
        candidate_weights = mat_e[:int(timesteps_matrix[f+1]) + 1, f]
        sum_weight = np.sum(candidate_weights)
        prob_sequence = candidate_weights / sum_weight
        cur_step = np.random.choice(range(0, int(timesteps_matrix[f+1]) + 1), 
                                    p=prob_sequence)
        timesteps_matrix[f] = int(cur_step)
    
    # get the timesteps from f+1 to F
    for f in range(curf + 1, F):
        candidate_weights = mat_s[int(timesteps_matrix[f-1]):, f]
        sum_weight = np.sum(candidate_weights)
        prob_sequence = candidate_weights / sum_weight
        cur_step = np.random.choice(range(int(timesteps_matrix[f-1]), self.T), 
                                    p=prob_sequence)
        timesteps_matrix[f] = int(cur_step)
    
    return timesteps_matrix


def get_noisy_model_input(T, F, ts, sigmas, x, noise):
    '''
    T: total time step for training
    F: total frame
    ts: sample timestep for batch
    '''
    noisy_model_input = x
    bt = []
    sigmas_t = []
    for i, one_batch_timestep in enumerate(ts):
        timesteps_matrix = (T, F, one_batch_timestep)
        bt.append(timesteps_matrix)
        for j, t in enumerate(timesteps_matrix):
            sigma = sigmas[t.long()]
            sigmas_t.append(sigma)
            noisy_model_input[i,:,j,:,:,] = noisy_model_input[i,:,j,:,:,] * (1.0 - sigma) + noise * sigma
    
    return noisy_model_input, bt, sigmas_t



def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
    set_seed(args.seed)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    #load model and scheduler
    noise_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=args.num_train_timesteps)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    device = "cuda"
    weight_dtype = torch.bfloat16
    vae_model_path = os.path.join(args.model_path, "Wan2.1_VAE.pth")
    vae = get_vae(vae_model_path, device, weight_dtype=torch.float32)
    text_encoder = get_text_encoder(model_path, device, weight_dtype)

    # load wan model for train
    config_path = os.path.join(args.dit_path, "config.json")
    wan_model = WanModel.from_config(config_path).to(torch.bfloat16).to("cuda")
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(args.dit_path, file)
            state_dict = load_file(file_path)
            wan_model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()
    if args.gradient_checkpointing:
        wan_model.enable_gradient_checkpointing()
    
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(wan_model).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(ltx_model).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    #if args.allow_tf32:
    #    torch.backends.cuda.matmul.allow_tf32 = True
    #    torch.backends.cudnn.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )      
    optimize_params = []
    for param in wab_model.parameters():
        if param.requires_grad:
            optimize_params.append(param)
    learnable_parameters_with_lr = {"params": optimize_params, "lr": args.learning_rate}
    params_to_optimize = [learnable_parameters_with_lr]
    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset
    train_dataset = BaseDataset(args)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    wan_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        wan_model, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    #start train df model
    video_processor = VideoProcessor(vae_scale_factor=16)
    for epoch in range(first_epoch, args.num_train_epochs):
        wan_model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(wan_model):
                with torch.no_grad():
                    latents = vae.encode().latent_dist.sample()
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                F = latents.shape[2]
                '''
                do not use normal distribution sample
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                '''
                timesteps = []
                for b in bsz:
                    indices = np.random.randint(0, args.num_train_timesteps)
                    timesteps.append(noise_scheduler_copy.timesteps[indices])
                sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
                noisy_model_input, bt = get_noisy_model_input(args.num_train_timesteps, F, timesteps, sigmas, latents, noise)
                prompt_embeds = text_encoder.encode(prompt).to(self.transformer.dtype)
                model_pred = wan_model(noisy_model_input[0], bt, prompt_embeds)
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas_t) + noisy_model_input
                loss = torch.mean(
                     (model_pred.float() - target.float() ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = optimize_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break
                







if __name__ == "__main__":
    args = parse_args()
    main(args)