from argparse import ArgumentParser
from transformers import SchedulerType

def get_train_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "-d", 
        "--dataset",
        type=str,
        default=None,
        help="path to the dataset to use to finetune the model"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt2",
        choices=[],
        help="What model should be finetuned"
    )


    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=2e-5,
        choices=[],
        help="What model should be finetuned"
    )

    parser.add_argument(
        "-wd",
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use.")

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help=""
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=5,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=3,
        help="the number of epochs to train the model for."
    )

    parser.add_argument(
        "--num_warmup_steps", 
        type=int, default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )

    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
        )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=32,
        help="The number of samples to include in each batch during training."
    )

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "-msl", 
        "--max_seq_len", 
        type=int, 
        default=500, 
        help="Maximum sequence length for the inputs."
    )
    parser.add_argument(
        "-ws", 
        "--warmup_steps", 
        type=int, 
        default=200, 
        help="Number of warmup steps for the scheduler."
    )

    parser.add_argument(
        "-od", 
        "--output_dir", 
        type=str, 
        default="checkpoints/", 
        help="Directory for saving model checkpoints."
    )
    parser.add_argument(
        "-op", 
        "--output_prefix", 
        type=str, 
        default="pynonedidit", 
        help="Prefix for the output files."
    )
    parser.add_argument(
        "-tm", 
        "--test_mode", 
        type=bool, 
        default=False, 
        help="Whether to run in test mode or not."
    )
    parser.add_argument(
        "-sm", 
        "--save_model_on_epoch", 
        type=bool, 
        default=True, 
        help="Whether to save the model after each epoch."
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    return parser