echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="/srv/share4/shalbe3/datasets/imagenet-r-train/"

accelerate launch textual_inversion_mixed.py  \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR   \
--learnable_property="object"  \
--resolution=512   \
--train_batch_size=4   \
--gradient_accumulation_steps=4   \
--max_train_steps=480000  \
--learning_rate=5.0e-04 \
--scale_lr   \
--lr_scheduler="constant"   \
--lr_warmup_steps=0   \
--output_dir="/srv/share4/shalbe3/text_inversion/checkpoints" \
--repeats 20 \
--use_auth_token