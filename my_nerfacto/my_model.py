"""
Template Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type


from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
#from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

@dataclass
class MyModelConfig(NerfactoModelConfig):
    """Template Model Configuration.
    
    Add your custom model config parameters here.
    """
    _target: Type = field(default_factory=lambda: MyNerfactoModel)
    eval_cam: bool = False
    """Whether to update eval camera poses during training"""


class MyNerfactoModel(NerfactoModel):
    """Nerfacto with eval cam pose optimization."""

    config: MyModelConfig

    def populate_modules(self):
        
        super().populate_modules()
        if self.config.eval_cam:
            self.eval_cam_cb = 0
            self.use_eval = False


    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
        ) -> List[TrainingCallback]:
            callbacks = []

            if self.config.eval_cam:
                def count_evalcam_cb(step):
                    if self.use_eval:
                        self.eval_cam_cb+=1
                    if self.eval_cam_cb%10==0:
                        self.use_eval = False

                    
                def eval_cam_opt(step):
                    
                    if step>0 and step%100==0:
                        self.use_eval=True
                    

                    if self.use_eval:
                        
                        print(self.eval_cam_cb)
                        for k, v in self.get_param_groups().items():
                            if k != 'camera_opt':
                                for p in v:
                                    p.requires_grad = False
                    
                    else: 
                        for k, v in self.get_param_groups().items():
                            if k != 'camera_opt':
                                for p in v:                                 
                                    p.requires_grad = True
                            
                    
                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        
                        func=eval_cam_opt,
                    )
                )

                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        
                        func=count_evalcam_cb,
                    )
                )
                    
            
            if self.config.use_proposal_weight_anneal:
                # anneal the weights of the proposal network before doing PDF sampling
                N = self.config.proposal_weights_anneal_max_num_iters

                def set_anneal(step):
                    # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                    self.step = step
                    train_frac = np.clip(step / N, 0, 1)
                    self.step = step

                    def bias(x, b):
                        return b * x / ((b - 1) * x + 1)

                    anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                    self.proposal_sampler.set_anneal(anneal)

                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=set_anneal,
                    )
                )
            
                
                callbacks.append(
                    TrainingCallback(
                        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                        update_every_num_iters=1,
                        func=self.proposal_sampler.step_cb,
                    )
                )
            return callbacks
    
    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        #if self.training: #should also be applied during eval(?)
        self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs
