import warnings
from typing import Optional, Dict, List, Type, Any

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric
import torch.nn.functional as F

from schnetpack.model.base import AtomisticModel

__all__ = ["ModelOutput_BNN", "AtomisticTask_BNN", "ModelOutput_MC",
           "AtomisticTask_MC", "ModelOutput_ENN", "AtomisticTask_ENN"
           ]

class ModelOutput_MC(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.train_metrics = nn.ModuleDict(metrics)
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])

class AtomisticTask_MC(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss

    ### 修正+新增：log_metrics（新增MC-Dropout不确定性日志，标量化处理） ###
    def log_metrics(self, pred, targets, subset,
                    mc_uncertainty_dict: Optional[Dict[str, Dict[str, torch.Tensor]]] = None):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=True,
                )

            ### 新增：MC-Dropout不确定性日志（标量化后再log） ###
            if mc_uncertainty_dict is not None and output.name in mc_uncertainty_dict:
                uncerts = mc_uncertainty_dict[output.name]
                # 标量化：批量均值转换为单元素张量
                pred_var_mean = torch.mean(uncerts["var"])
                pred_std_mean = torch.mean(uncerts["std"])

                # 日志记录不确定性（方差+标准差）
                self.log(
                    f"{subset}_{output.name}_mc_variance",
                    pred_var_mean,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )
                self.log(
                    f"{subset}_{output.name}_mc_std",
                    pred_std_mean,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=True,  # 进度条显示标准差，直观查看不确定性
                )

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        mc_uncertainty_dict = {}
        try:
            mc_uncertainty = self.model.output_modules.get_cached_uncertainty()
            # 假设输出名与模型output_key一致，可根据实际情况调整
            for output in self.outputs:
                mc_uncertainty_dict[output.name] = mc_uncertainty
        except RuntimeError:
            warnings.warn("蒙特卡洛采样未完成，跳过不确定性日志记录！")

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log_metrics(pred, targets, "train", mc_uncertainty_dict=mc_uncertainty_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        mc_uncertainty_dict = {}
        try:
            mc_uncertainty = self.model.output_modules.get_cached_uncertainty()
            # 假设输出名与模型output_key一致，可根据实际情况调整
            for output in self.outputs:
                mc_uncertainty_dict[output.name] = mc_uncertainty
        except RuntimeError:
            warnings.warn("蒙特卡洛采样未完成，跳过不确定性日志记录！")

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "val", mc_uncertainty_dict=mc_uncertainty_dict)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        mc_uncertainty_dict = {}
        try:
            mc_uncertainty = self.model.output_modules.get_cached_uncertainty()
            # 假设输出名与模型output_key一致，可根据实际情况调整
            for output in self.outputs:
                mc_uncertainty_dict[output.name] = mc_uncertainty
        except RuntimeError:
            warnings.warn("蒙特卡洛采样未完成，跳过不确定性日志记录！")

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "test", mc_uncertainty_dict=mc_uncertainty_dict)
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate scheduling is set to occur after the epoch ends. To enable scheduling before the "
                    "epoch end, please set the `val_check_interval` parameter to a value greater than 1.0, which "
                    "indicates the number of training steps after which the model should be validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing
            torch.save(self.model, path)
            self.model.do_postprocessing = pp_status


class ModelOutput_BNN(nn.Module):
    """
    定义模型输出，兼容贝叶斯神经网络，修复潜在报错，支持损失标量化
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        # 修复：处理metrics为None的情况，避免报错
        self.train_metrics = nn.ModuleDict(metrics) if metrics is not None else nn.ModuleDict()
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in self.train_metrics.items()}) if metrics is not None else nn.ModuleDict()
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in self.train_metrics.items()}) if metrics is not None else nn.ModuleDict()
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )

        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])


class AtomisticTask_BNN(pl.LightningModule):
    """
    兼容贝叶斯神经网络的AtomisticTask，支持证据下界（ELBO）损失计算和不确定性日志
    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
        bnn_beta: float = 1.0,  # BNN的KL散度权重（ELBO损失：likelihood_loss + beta * KL_div）
    ):
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args or {}  # 修复：处理optimizer_args为None的情况
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args or {}  # 修复：处理scheduler_args为None的情况
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)
        self.bnn_beta = bnn_beta  # BNN专属：KL散度权重

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = self.optimizer_kwargs.get("lr", 1e-3)  # 修复：避免KeyError
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        # 计算普通预测损失（似然损失）
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)

        # BNN专属：添加KL散度正则项（构成ELBO损失）
        try:
            kl_div = self.model.output_modules.get_cached_kl_div()
            # KL散度通常需要除以批量大小，保证训练稳定性
            batch_size = len(batch["_idx"]) if "_idx" in batch else 1
            elbo_kl_loss = self.bnn_beta * (kl_div / batch_size)
            loss += elbo_kl_loss

            # 缓存KL散度损失，用于日志记录
            self.kl_loss_cache = elbo_kl_loss
        except RuntimeError:
            warnings.warn("BNN KL散度缓存未生成，跳过KL正则项！")
            self.kl_loss_cache = torch.tensor(0.0, device=loss.device)

        return loss

    ### 修正：log_metrics（新增BNN不确定性日志和KL散度日志） ###
    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=True,
                )

        # BNN专属：记录KL散度损失
        if hasattr(self, "kl_loss_cache"):
            self.log(
                f"{subset}_bnn_kl_loss",
                self.kl_loss_cache,
                on_step=(subset == "train"),
                on_epoch=(subset != "train"),
                prog_bar=True,
            )

    # BNN专属：记录不确定性指标（均值+标准差）
        try:
            uncertainty = self.model.output_modules.get_cached_uncertainty()
            # 标量化后再日志，避免非单元素张量报错
            pred_std_mean = torch.mean(uncertainty["std"])
            self.log(
                f"{subset}_bnn_pred_std",
                pred_std_mean,
                on_step=(subset == "train"),
                on_epoch=(subset != "train"),
                prog_bar=True,
            )
        except RuntimeError:
            warnings.warn("BNN不确定性缓存未生成，跳过不确定性日志！")

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    ### 修正：training_step（兼容BNN，损失标量化） ###
    def training_step(self, batch, batch_idx):
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except KeyError:  # 修复：精确异常捕获
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)


        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")
        return loss

    ### 修正：validation_step（兼容BNN，损失标量化） ###
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except KeyError:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)


        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "val")

        return loss

    ### 修正：test_step（兼容BNN，损失标量化） ###
    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except KeyError:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)


        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "test")
        return loss

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing

            torch.save(self.model, path)

            self.model.do_postprocessing = pp_status


class ModelOutput_ENN(nn.Module):
    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        reg_coeff: float = 0.1,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
        ### 新增：标记是否为证据输出（用于切换损失输入） ###
        is_evidential: bool = True,
    ):
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.train_metrics = nn.ModuleDict(metrics) if metrics is not None else nn.ModuleDict()
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in self.train_metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in self.train_metrics.items()})
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.constraints = constraints or []
        ### 新增：记录是否为证据输出 ###
        self.is_evidential = is_evidential

    ### 修正：新增nig_params参数传入，不依赖pred字典获取NIG参数 ###
    def calculate_loss(self, pred, target, nig_params: Optional[torch.Tensor] = None):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        # 普通输出：使用pred中的预测值
        if not self.is_evidential:
            loss_input = pred[self.name]
        # 证据输出：使用传入的nig_params（不依赖pred字典）
        else:
            if nig_params is None:
                raise RuntimeError("对于证据输出，必须传入nig_params参数！")
            loss_input = nig_params

        loss = self.loss_weight * self.loss_fn(
            loss_input, target[self.target_property]
        )
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])

class AtomisticTask_ENN(pl.LightningModule):
    """
    基础学习任务，通过模型缓存获取NIG参数并传递给损失函数，不修改pred字典
    """
    def __init__(
        self,
        model: AtomisticModel,
        outputs,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args or {}
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = self.optimizer_kwargs.get("lr", 1e-3)
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    ### 修正：新增nig_params_dict参数，传递各输出对应的NIG参数 ###
    def loss_fn(self, pred, target, nig_params_dict: Optional[Dict[str, torch.Tensor]] = None):
        loss = 0.0
        nig_params_dict = nig_params_dict or {}
        for output in self.outputs:
            # 对于证据输出，从nig_params_dict获取对应NIG参数
            if output.is_evidential:
                nig_params = nig_params_dict.get(output.name, None)
                loss += output.calculate_loss(pred, target, nig_params=nig_params)
            # 普通输出，正常计算
            else:
                loss += output.calculate_loss(pred, target)
        return loss

    ### 修正：新增uncertainty_dict参数，传递不确定性用于日志记录 ###
    def log_metrics(self, pred, targets, subset, uncertainty_dict: Optional[Dict[str, Dict[str, torch.Tensor]]] = None):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=True,
                )
            ### 记录证据神经网络的不确定性指标（使用传入的uncertainty_dict） ###
            if output.is_evidential and uncertainty_dict is not None and output.name in uncertainty_dict:
                uncerts = uncertainty_dict[output.name]
                epistemic_uncert = uncerts["epistemic"]
                aleatoric_uncert = uncerts["aleatoric"]
                total_uncert = uncerts["total"]

                self.log(
                    f"{subset}_{output.name}_epistemic_uncertainty",
                    torch.mean(epistemic_uncert),
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )
                self.log(
                    f"{subset}_{output.name}_aleatoric_uncertainty",
                    torch.mean(aleatoric_uncert),
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )
                self.log(
                    f"{subset}_{output.name}_total_uncertainty",
                    torch.mean(total_uncert),
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=True,
                )

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    ### 修正：训练步骤 - 从模型缓存获取NIG参数和不确定性，传递给损失/日志 ###
    def training_step(self, batch, batch_idx):
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except KeyError:
            pass

        # 1. 预测（此时模型已缓存NIG参数和不确定性）
        pred = self.predict_without_postprocessing(batch)
        # 2. 应用约束（pred仅保留{output.name}，不修改）
        pred, targets = self.apply_constraints(pred, targets)

        # 3. ### 新增：从模型缓存提取NIG参数和不确定性 ###
        nig_params_dict = {}
        uncertainty_dict = {}
        # 遍历输出，获取对应ENN模型的缓存
        for output in self.outputs:
            if output.is_evidential:
                try:
                    nig_params = self.model.output_modules.get_cached_nig_params()
                    uncertainty = self.model.output_modules.get_cached_uncertainty()
                    nig_params_dict[output.name] = nig_params
                    uncertainty_dict[output.name] = uncertainty
                except RuntimeError:
                    warnings.warn(f"无法获取{output.name}对应的NIG参数缓存，请检查模型forward是否已调用！")

        # 4. ### 修正：传入nig_params_dict计算损失 ###
        loss = self.loss_fn(pred, targets, nig_params_dict=nig_params_dict)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        # 5. ### 修正：传入uncertainty_dict记录日志 ###
        self.log_metrics(pred, targets, "train", uncertainty_dict=uncertainty_dict)
        return loss

    ### 修正：验证步骤 - 与训练步骤逻辑一致 ###
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except KeyError:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        # 提取NIG参数和不确定性缓存
        nig_params_dict = {}
        uncertainty_dict = {}
        for output in self.outputs:
            if output.is_evidential:
                try:
                    nig_params = self.model.output_modules.get_cached_nig_params()
                    uncertainty = self.model.output_modules.get_cached_uncertainty()
                    nig_params_dict[output.name] = nig_params
                    uncertainty_dict[output.name] = uncertainty
                except RuntimeError:
                    warnings.warn(f"无法获取{output.name}对应的NIG参数缓存，请检查模型forward是否已调用！")

        loss = self.loss_fn(pred, targets, nig_params_dict=nig_params_dict)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "val", uncertainty_dict=uncertainty_dict)

        return {"val_loss": loss}

    ### 修正：测试步骤 - 与训练步骤逻辑一致 ###
    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except KeyError:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        # 提取NIG参数和不确定性缓存
        nig_params_dict = {}
        uncertainty_dict = {}
        for output in self.outputs:
            if output.is_evidential:
                try:
                    nig_params = self.model.output_modules.get_cached_nig_params()
                    uncertainty = self.model.output_modules.get_cached_uncertainty()
                    nig_params_dict[output.name] = nig_params
                    uncertainty_dict[output.name] = uncertainty
                except RuntimeError:
                    warnings.warn(f"无法获取{output.name}对应的NIG参数缓存，请检查模型forward是否已调用！")

        loss = self.loss_fn(pred, targets, nig_params_dict=nig_params_dict)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["_idx"]),
        )
        self.log_metrics(pred, targets, "test", uncertainty_dict=uncertainty_dict)
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)  # 调用forward后，模型已缓存NIG参数和不确定性
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate scheduling is set to occur after the epoch ends. To enable scheduling before the "
                    "epoch end, please set the `val_check_interval` parameter to a value greater than 1.0, which "
                    "indicates the number of training steps after which the model should be validated."
                )
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing
            torch.save(self.model, path)
            self.model.do_postprocessing = pp_status


class EvidentialLoss(nn.Module):
    """
    回归任务的证据损失函数（Evidential MSE Loss）
    核心：MSE损失 + 证据正则项，约束NIG参数的合理性，避免模型过度自信
    """

    def __init__(self, reg_coeff: float = 0.1):
        super().__init__()
        self.reg_coeff = reg_coeff  # 正则项系数，可调

    def forward(self, nig_params: torch.Tensor, target: torch.Tensor):
        # 解析NIG参数：mu(均值), v(证据量), alpha(形状), beta(尺度)
        mu, v, alpha, beta = torch.split(nig_params, 1, dim=-1)

        # 约束参数范围（确保物理意义，避免数值不稳定）
        v = F.softplus(v) + 1e-6  # 证据量v > 0
        alpha = F.softplus(alpha) + 1.0 + 1e-6  # alpha > 1
        beta = F.softplus(beta) + 1e-6  # 尺度beta > 0
        target = target.unsqueeze(-1)  # 维度对齐

        # 1. MSE损失项（针对预测均值mu）
        mse_loss = torch.mean((mu - target) ** 2)

        # 2. 证据正则项（鼓励模型在不确定时增加证据，防止过度自信）
        reg_term = torch.mean(2 * beta * (1 + v) + v * (mu - target) ** 2) / (2 * (alpha - 1))
        reg_term = torch.mean(reg_term)
        # 总损失：MSE + 正则项
        total_loss = mse_loss + self.reg_coeff * reg_term
        return total_loss

class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.train_metrics = nn.ModuleDict(metrics)
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])


class UnsupervisedModelOutput(ModelOutput):
    """
    Defines an unsupervised output of a model, i.e. an unsupervised loss or a regularizer
    that do not depend on label data. It includes mappings to the loss function,
    a weight for training and metrics to be logged.
    """

    def calculate_loss(self, pred, target=None):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(pred[self.name])
        return loss

    def update_metrics(self, pred, target, subset):
        for metric in self.metrics[subset].values():
            metric(pred[self.name])


class AtomisticTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),
                    on_epoch=(subset != "train"),
                    prog_bar=False,
                )

    def apply_constraints(self, pred, targets):
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "test")
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        pp = self.model.do_postprocessing
        self.model.do_postprocessing = False
        pred = self(batch)
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )

        if self.scheduler_cls:
            schedulers = []
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            if self.trainer.val_check_interval < 1.0:
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            schedulers.append(optimconf)
            return [optimizer], schedulers
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        if self.global_step < self.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        if self.global_rank == 0:
            pp_status = self.model.do_postprocessing
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing

            torch.save(self.model, path)

            self.model.do_postprocessing = pp_status