"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from my_nerfacto.my_datamanager import (
    TemplateDataManagerConfig,
)
from my_nerfacto.my_model import MyModelConfig
from my_nerfacto.my_pipeline import (
    TemplatePipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


MyMethod = MethodSpecification(
    config=TrainerConfig(
        method_name="my_nerfacto",  
        steps_per_eval_batch=500,
        steps_per_eval_all_images=500,
        steps_per_save=2000,
        max_num_iterations=2000,
        mixed_precision=True,
        pipeline=TemplatePipelineConfig(
            datamanager=TemplateDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=MyModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
                eval_cam=True
                
                
            ),
        ),
        optimizers={
            # TODO: consider changing optimizers depending on your custom method
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio method template.",
)