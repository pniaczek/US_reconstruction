#!/usr/bin/env bash
set -e

cd deep-spectral-segmentation

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_API_KEY=

export WANDB_CONFIG_DIR=/tmp/
export WANDB_CACHE_DIR=/tmp/
export WANDB_AGENT_MAX_INITIAL_FAILURE=20
export WANDB__SERVICE_WAIT=600
export XFORMERS_DISABLED=True

python -m pipeline.pipeline_sweep_subfolders \
    vis=selected \
    pipeline_steps=defaults \
    pipeline_steps.crf_segm=true \
    pipeline_steps.crf_multi_region=false \
    dataset=hamzaoran_unsup_tv_seg50_clust15 \
    wandb=defaults \
    wandb.tag=tv_seg50_clust15 \
    sweep=hamzaoran_unsup_tv_seg50_clust15_clusters
