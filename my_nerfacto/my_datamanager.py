"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, List

import torch

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)


@dataclass
class TemplateDataManagerConfig(VanillaDataManagerConfig):
    """Template DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TemplateDataManager)


class TemplateDataManager(VanillaDataManager):
    """Template DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TemplateDataManagerConfig

    def __init__(
        self,
        config: TemplateDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        
        self.next_train_cache = self.next_train

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if training_callback_attributes.pipeline.model.config.eval_cam:
            def use_eval_cams(step):
                if training_callback_attributes.pipeline.model.use_eval:

                    self.next_train = self.next_eval
                    print('eval_cam_opt')
                else:
                    self.next_train = self.next_train_cache    
        

            callbacks.append(
                        TrainingCallback(
                            where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                            update_every_num_iters=1,
                            
                            func=use_eval_cams,
                        )
                    )
                     
        """Returns a list of callbacks to be used during training."""

        return callbacks


    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        #print(self.train_count)
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch