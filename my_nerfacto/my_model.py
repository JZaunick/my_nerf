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
import math
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
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer, SemanticRenderer
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

    eval_cam: bool = True
    """Whether to update eval camera poses during training"""
    eval_cam_interval = 100
    """after how many steps to update eval cam poses"""
    eval_cam_steps = 10
    """for how many steps to update cam poses"""
    


class MyNerfactoModel(NerfactoModel):
    """Nerfacto with eval cam pose optimization."""

    config: MyModelConfig




    def populate_modules(self):
        
        
        super().populate_modules()
        
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0


           # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

        #get number of eval data
        self.num_eval_data = math.floor((self.num_train_data/0.9-self.num_train_data)) #TODO: train fraction should not be hardcoded
        print('number of eval data:', self.num_eval_data)
        
        #cam optimizers for train and eval dataset 
        self.camera_optimizer_train: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=(self.num_train_data), device="cpu"
        )
        
        self.camera_optimizer_eval: CameraOptimizer = self.config.camera_optimizer.setup(
                        num_cameras=(self.num_eval_data), device="cpu"
                        )
        

        self.camera_optimizer = self.camera_optimizer_train

        

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
        ) -> List[TrainingCallback]:
            callbacks = []

            if self.config.eval_cam:

                                   
                def eval_cam_opt(step):
                    
               
                    
                    #switch to eval-pose.optimization:
                    if step % self.config.eval_cam_interval <  self.config.eval_cam_steps and step >= self.config.eval_cam_interval:
                        #use eval set
                        training_callback_attributes.pipeline.datamanager.next_train = training_callback_attributes.pipeline.datamanager.next_eval
                                               
                        #re-setup optimizers in first step of interval
                        if step % self.config.eval_cam_interval == 0 and step > 0:
                            #print(f"Condition met at step {step}")
                            
                            self.camera_optimizer = self.camera_optimizer_eval
                            #cache train optimizers and re-setup for eval dataset:
                            self.cached_train_optimizers = training_callback_attributes.trainer.optimizers
                            training_callback_attributes.trainer.optimizers= training_callback_attributes.trainer.setup_optimizers()
                            
                        for k, v in self.get_param_groups().items():
                            if k != 'camera_opt':
                                for p in v:
                                    p.requires_grad = False
                    
                   
                    
                    
                    else:
                        training_callback_attributes.pipeline.datamanager.next_train = training_callback_attributes.pipeline.datamanager.next_train_cache
                        

                        #switch back to train dataset optimizers after using eval cam poses
                        if step %self.config.eval_cam_interval==self.config.eval_cam_steps and step >= self.config.eval_cam_interval:                       
                            self.camera_optimizer = self.camera_optimizer_train
                            training_callback_attributes.trainer.optimizers=self.cached_train_optimizers
                            
                        
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
        if self.training: 
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
        #semantic = self.renderer_semantics(semantics=field_outputs[FieldHeadNames.RGB],weights=weights)

        

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            #"semantics": semantic
        }

        # semantics colormaps
        #outputs['semantic_labels'] = torch.argmax(torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1)
        #outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]


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

