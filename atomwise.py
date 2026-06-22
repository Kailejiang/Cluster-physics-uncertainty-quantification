from typing import Sequence, Union, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import pandas as pd
import schnetpack as spk
import schnetpack.nn as snn
import schnetpack.properties as properties

__all__ = ["DipoleMoment", "Polarizability","Atomwise", 
           "MC_Dropout", "EvidentialNN", "BayesianNN"]

class EvidentialNN(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
    ):
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out
        self.nig_params_cache = None  # ### 新增：缓存NIG参数的变量 ###
        self.uncertainty_cache = None  # ### 新增：缓存不确定性的变量 ###
        self.tsne_features = []
        self.tsne_energies = []
        self.tsne_uncertainties = []

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_enn_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        out_nur = n_in//2
        self.nig_head = nn.Linear(out_nur, 4)
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 重置缓存
        self.nig_params_cache = None
        self.uncertainty_cache = None

        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        ### 新增：生成并缓存NIG参数（不写入最终inputs/pred） ###
        nig_params = self.nig_head(y)
        self.nig_params_cache = nig_params  # 缓存NIG参数，供后续损失计算使用

        ### 新增：计算并缓存不确定性（不写入最终inputs/pred） ###
        mu, v, alpha, beta = torch.split(nig_params, self.n_out, dim=-1)
        # 约束参数范围
        v = F.softplus(v) + 1e-6
        alpha = F.softplus(alpha) + 1.0 + 1e-6
        beta = F.softplus(beta) + 1e-6
        # 计算不确定性
        epistemic_uncertainty = beta / (alpha - 1) / v
        aleatoric_uncertainty = beta / (alpha - 1)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        # 缓存不确定性
        self.uncertainty_cache = {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
            "total": total_uncertainty
        }

        ### 仅保留{output_key}在inputs中（最终pred仅含此项，不修改pred结构） ###
        mu, _, _, _ = torch.split(nig_params, self.n_out, dim=-1)
        inputs[self.output_key] = mu.squeeze(-1) if self.n_out == 1 else mu

        if not self.training:
            with torch.no_grad():

                # 1️⃣ 原子特征
                atom_feat = inputs["scalar_representation"]

                # 2️⃣ 聚合为分子级特征
                idx_m = inputs[properties.idx_m]
                maxm = int(idx_m[-1]) + 1

                mol_feat = snn.scatter_add(atom_feat, idx_m, dim_size=maxm)

                if self.aggregation_mode == "avg":
                    mol_feat = mol_feat / inputs[properties.n_atoms]

                # 3️⃣ energy（μ）
                energy = inputs[self.output_key]

                # 4️⃣ uncertainty（用 total）
                if self.uncertainty_cache is not None:
                    uncertainty = self.uncertainty_cache["total"]
                else:
                    uncertainty = torch.zeros_like(energy)

                # 5️⃣ 强制转CPU（防炸）
                self.tsne_features.append(mol_feat.detach().to("cpu"))
                self.tsne_energies.append(energy.detach().to("cpu"))
                self.tsne_uncertainties.append(uncertainty.detach().to("cpu"))

        return inputs

    ### 新增：获取缓存的NIG参数（外部调用，不修改pred） ###
    def get_cached_nig_params(self):
        if self.nig_params_cache is None:
            raise RuntimeError("请先调用forward方法生成NIG参数后，再获取缓存！")
        return self.nig_params_cache

    ### 新增：获取缓存的不确定性（外部调用，不修改pred） ###
    def get_cached_uncertainty(self):
        if self.uncertainty_cache is None:
            raise RuntimeError("请先调用forward方法生成不确定性后，再获取缓存！")
        return self.uncertainty_cache

    def export_tsne_csv(self, save_path="tsne_results.csv"):
        """
        对缓存的分子特征做t-SNE降维，并导出CSV
        """

        if len(self.tsne_features) == 0:
            raise RuntimeError("没有收集到任何特征，请先运行eval forward！")

        import torch
        import pandas as pd
        from sklearn.manifold import TSNE

        # ✅ 强制统一CPU（避免你之前那个bug）
        X = torch.cat([t.cpu() for t in self.tsne_features], dim=0).numpy()
        y = torch.cat([t.cpu() for t in self.tsne_energies], dim=0).numpy()
        u = torch.cat([t.cpu() for t in self.tsne_uncertainties], dim=0).numpy()

        print(f"t-SNE输入维度: {X.shape}")

        # ================== t-SNE ==================
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            random_state=42
        )

        X_embedded = tsne.fit_transform(X)

        # ================== CSV ==================
        df = pd.DataFrame({
            "dim1": X_embedded[:, 0],
            "dim2": X_embedded[:, 1],
            "uncertainty": u.flatten(),
            "energy": y.flatten()
        })

        df.to_csv(save_path, index=False)

        print(f"t-SNE结果已保存到: {save_path}")

    def reset_tsne_cache(self):
        self.tsne_features = []
        self.tsne_energies = []
        self.tsne_uncertainties = []
    
class MC_Dropout(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
        mc_samples: int = 50,
    ):
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out
        self.mc_samples = mc_samples  # 保存采样次数
        self.mc_predictions_cache = None  # ### 新增：缓存蒙特卡洛采样结果 ###
        self.uncertainty_cache = None  # ### 新增：缓存不确定性（均值+方差） ###

        self.tsne_features = []
        self.tsne_energies = []
        self.tsne_uncertainties = []

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_dropout_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        inputs[self.output_key] = y

        ### 新增：MC-Dropout核心逻辑 - 蒙特卡洛采样（训练/验证/测试时启用dropout） ###
        if self.training or (not self.training and self.mc_samples > 1):
            # 保存多次采样结果：形状 [mc_samples, batch_size, n_out]
            mc_predictions = []
            # 开启dropout（评估模式下强制启用dropout层）
            self.outnet.train()
            for _ in range(self.mc_samples):
                # 重复前向传播（原子级预测 + 聚合）
                y_mc = self.outnet(inputs["scalar_representation"])
                if self.aggregation_mode is not None:
                    y_mc = snn.scatter_add(y_mc, idx_m, dim_size=maxm)
                    y_mc = torch.squeeze(y_mc, -1)
                    if self.aggregation_mode == "avg":
                        y_mc = y_mc / inputs[properties.n_atoms]
                mc_predictions.append(y_mc.unsqueeze(0))  # 增加采样维度
            # 拼接采样结果并缓存
            mc_predictions = torch.cat(mc_predictions, dim=0)
            self.mc_predictions_cache = mc_predictions

            ### 新增：计算不确定性（均值=点预测，方差=不确定性度量） ###
            pred_mean = torch.mean(mc_predictions, dim=0)  # 采样均值（更稳健的点预测）
            pred_var = torch.var(mc_predictions, dim=0)    # 采样方差（不确定性大小）
            # 缓存不确定性（均值+方差）
            self.uncertainty_cache = {
                "mean": pred_mean,    # 蒙特卡洛均值
                "var": pred_var,      # 蒙特卡洛方差（不确定性）
                "std": torch.sqrt(pred_var + 1e-6)  # 标准差（可选，更易解释）
            }

        if not self.training:
            with torch.no_grad():

                # 1️⃣ 原子特征
                atom_feat = inputs["scalar_representation"]

                # 2️⃣ 聚合为分子级特征
                idx_m = inputs[properties.idx_m]
                maxm = int(idx_m[-1]) + 1

                mol_feat = snn.scatter_add(atom_feat, idx_m, dim_size=maxm)

                if self.aggregation_mode == "avg":
                    mol_feat = mol_feat / inputs[properties.n_atoms]

                # 3️⃣ energy（用MC均值更合理）
                if self.uncertainty_cache is not None:
                    energy = self.uncertainty_cache["mean"]
                    uncertainty = self.uncertainty_cache["std"]
                else:
                    energy = inputs[self.output_key]
                    uncertainty = torch.zeros_like(energy)

                # 4️⃣ 强制转CPU（避免你之前那个报错）
                self.tsne_features.append(mol_feat.detach().to("cpu"))
                self.tsne_energies.append(energy.detach().to("cpu"))
                self.tsne_uncertainties.append(uncertainty.detach().to("cpu"))

        return inputs

    def export_tsne_csv(self, save_path="tsne_results.csv"):
        if len(self.tsne_features) == 0:
            raise RuntimeError("没有收集到任何特征，请先运行eval forward！")

        import torch
        import pandas as pd
        from sklearn.manifold import TSNE

        # ✅ 强制统一CPU（彻底避免device bug）
        X = torch.cat([t.cpu() for t in self.tsne_features], dim=0).numpy()
        y = torch.cat([t.cpu() for t in self.tsne_energies], dim=0).numpy()
        u = torch.cat([t.cpu() for t in self.tsne_uncertainties], dim=0).numpy()

        print(f"t-SNE输入维度: {X.shape}")

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            random_state=42
        )

        X_embedded = tsne.fit_transform(X)

        df = pd.DataFrame({
            "dim1": X_embedded[:, 0],
            "dim2": X_embedded[:, 1],
            "uncertainty": u.flatten(),
            "energy": y.flatten()
        })

        df.to_csv(save_path, index=False)

        print(f"t-SNE结果已保存到: {save_path}")

    def reset_tsne_cache(self):
        self.tsne_features = []
        self.tsne_energies = []
        self.tsne_uncertainties = []

    def get_cached_mc_predictions(self):
        if self.mc_predictions_cache is None:
            raise RuntimeError("请先调用forward方法完成蒙特卡洛采样后，再获取缓存！")
        return self.mc_predictions_cache

    ### 新增：获取不确定性缓存 ###
    def get_cached_uncertainty(self):
        if self.uncertainty_cache is None:
            raise RuntimeError("请先调用forward方法完成蒙特卡洛采样后，再获取不确定性缓存！")
        return self.uncertainty_cache


class BayesianNN(nn.Module):
    """
    贝叶斯版本的Atomwise模型，预测原子级贡献并聚合为全局预测，支持不确定性量化
    """
    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        mc_samples: int = 50,  # 蒙特卡洛采样次数（用于不确定性量化）
    ):
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.mc_samples = mc_samples
        self.kl_div_cache = None  # 缓存KL散度（贝叶斯正则项）
        self.mc_predictions_cache = None  # 缓存蒙特卡洛采样结果
        self.uncertainty_cache = None  # 缓存不确定性（均值+方差）
        self.tsne_features = []
        self.tsne_energies = []
        self.tsne_uncertainties = []

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_bayesian_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode


    def _compute_total_kl_div(self) -> torch.Tensor:
        """计算模型所有BayesianDense层的KL散度之和"""
        total_kl = 0.0
        for module in self.outnet.modules():
            if isinstance(module, snn.BayesianDense):
                total_kl += module.kl_divergence()
        return total_kl

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 重置缓存
        self.kl_div_cache = None
        self.mc_predictions_cache = None
        self.uncertainty_cache = None

        # 计算并缓存KL散度（贝叶斯模型的正则项，训练时需要）
        self.kl_div_cache = self._compute_total_kl_div()

        # 普通前向传播：单次采样的原子级预测
        y = self.outnet(inputs["scalar_representation"])

        # 保存原子级预测结果（若需要）
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # 聚合为全局预测
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        # 保存单次采样的全局预测结果
        inputs[self.output_key] = y

        ### 贝叶斯核心：蒙特卡洛采样量化不确定性（训练/评估模式均启用） ###
        if self.mc_samples > 1:
            mc_predictions = []
            # 多次采样获取预测分布
            for _ in range(self.mc_samples):
                # 原子级预测（重新采样权重）
                y_mc = self.outnet(inputs["scalar_representation"])
                # 聚合为全局预测
                if self.aggregation_mode is not None:
                    y_mc = snn.scatter_add(y_mc, idx_m, dim_size=maxm)
                    y_mc = torch.squeeze(y_mc, -1)
                    if self.aggregation_mode == "avg":
                        y_mc = y_mc / inputs[properties.n_atoms]
                mc_predictions.append(y_mc.unsqueeze(0))
            # 拼接采样结果并缓存
            mc_predictions = torch.cat(mc_predictions, dim=0)
            self.mc_predictions_cache = mc_predictions

            # 计算不确定性（均值=稳健点预测，方差=认知不确定性）
            pred_mean = torch.mean(mc_predictions, dim=0)
            pred_var = torch.var(mc_predictions, dim=0)
            self.uncertainty_cache = {
                "mean": pred_mean,
                "var": pred_var,
                "std": torch.sqrt(pred_var + 1e-6)  # 标准差，避免根号内为负
            }

        if not self.training:
            with torch.no_grad():
                # 1️⃣ 获取原子特征
                atom_feat = inputs["scalar_representation"]

                # 2️⃣ 聚合为分子级特征
                idx_m = inputs[properties.idx_m]
                maxm = int(idx_m[-1]) + 1

                mol_feat = snn.scatter_add(atom_feat, idx_m, dim_size=maxm)

                if self.aggregation_mode == "avg":
                    mol_feat = mol_feat / inputs[properties.n_atoms]

                # 3️⃣ 获取预测值 & 不确定性
                energy = inputs[self.output_key]

                if self.uncertainty_cache is not None:
                    uncertainty = self.uncertainty_cache["std"]
                else:
                    uncertainty = torch.zeros_like(energy)

                # 4️⃣ 存入缓存（转cpu防止显存爆）
                self.tsne_features.append(mol_feat.detach().cpu())
                self.tsne_energies.append(energy.detach().cpu())
                self.tsne_uncertainties.append(uncertainty.detach().cpu())

        return inputs

    def export_tsne_csv(self, save_path="tsne_results.csv"):
        """
        对缓存的分子特征做t-SNE降维，并导出CSV
        """
        if len(self.tsne_features) == 0:
            raise RuntimeError("没有收集到任何特征，请先运行eval forward！")

        # 拼接所有batch
        X = torch.cat([t.cpu() for t in self.tsne_features], dim=0).numpy()
        y = torch.cat([t.cpu() for t in self.tsne_energies], dim=0).numpy()
        u = torch.cat([t.cpu() for t in self.tsne_uncertainties], dim=0).numpy()

        print(f"t-SNE输入维度: {X.shape}")

        # ================== t-SNE降维 ==================
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            random_state=42
        )

        X_embedded = tsne.fit_transform(X)

        # ================== 保存CSV ==================
        df = pd.DataFrame({
            "dim1": X_embedded[:, 0],
            "dim2": X_embedded[:, 1],
            "uncertainty": u.flatten(),
            "energy": y.flatten()
        })

        df.to_csv(save_path, index=False)

        print(f"t-SNE结果已保存到: {save_path}")

    def reset_tsne_cache(self):
        self.tsne_features = []
        self.tsne_energies = []
        self.tsne_uncertainties = []

    ### 新增：获取缓存的KL散度（用于ELBO损失计算） ###
    def get_cached_kl_div(self) -> torch.Tensor:
        if self.kl_div_cache is None:
            raise RuntimeError("请先调用forward方法后，再获取KL散度缓存！")
        return self.kl_div_cache

    ### 新增：获取蒙特卡洛采样结果缓存 ###
    def get_cached_mc_predictions(self) -> torch.Tensor:
        if self.mc_predictions_cache is None:
            raise RuntimeError("请先调用forward方法并设置mc_samples>1后，再获取采样缓存！")
        return self.mc_predictions_cache

    ### 新增：获取不确定性缓存 ###
    def get_cached_uncertainty(self) -> Dict[str, torch.Tensor]:
        if self.uncertainty_cache is None:
            raise RuntimeError("请先调用forward方法并设置mc_samples>1后，再获取不确定性缓存！")
        return self.uncertainty_cache
    

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(Atomwise, self).__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        inputs[self.output_key] = y
        return inputs

class DipoleMoment(nn.Module):
    """
    Predicts dipole moments from latent partial charges and (optionally) local atomic dipoles.
    The latter requires an equivariant representation supplying vector features.

    References:
    .. [#painn1] Schütt, Unke, Gastegger.
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    .. [#irspec] Gastegger, Behler, Marquetand.
       Machine learning molecular dynamics for the simulation of infrared spectra.
       Chemical science 8.10 (2017): 6924-6935.
    .. [#dipole] Veit et al.
       Predicting molecular dipole moments by combining atomic partial charges and atomic dipoles.
       The Journal of Chemical Physics 153.2 (2020): 024113.
    """

    def __init__(
            self,
            n_in: int,
            n_hidden: Optional[Union[int, Sequence[int]]] = None,
            n_layers: int = 2,
            activation: Callable = F.silu,
            predict_magnitude: bool = False,
            return_charges: bool = False,
            dipole_key: str = properties.dipole_moment,
            charges_key: str = properties.partial_charges,
            correct_charges: bool = True,
            use_vector_representation: bool = False,
    ):
        super().__init__()

        self.dipole_key = dipole_key
        self.charges_key = charges_key
        self.return_charges = return_charges
        self.model_outputs = [dipole_key]
        if self.return_charges:
            self.model_outputs.append(charges_key)

        self.predict_magnitude = predict_magnitude
        self.use_vector_representation = use_vector_representation
        self.correct_charges = correct_charges

        # Build appropriate network based on vector representation usage
        if use_vector_representation:
            self.outnet = spk.nn.build_gated_equivariant_mlp(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                sactivation=activation,
            )
        else:
            self.outnet = spk.nn.build_mlp(
                n_in=n_in,
                n_out=1,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        natoms = inputs[properties.n_atoms]
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        # Predict charges and atomic dipoles (if using vector features)
        if self.use_vector_representation:
            l1 = inputs["vector_representation"]
            charges, atomic_dipoles = self.outnet((l0, l1))
            atomic_dipoles = torch.squeeze(atomic_dipoles, -1)
        else:
            charges = self.outnet(l0)
            atomic_dipoles = 0.0

        # Correct partial charges to match total charge (if enabled)
        if self.correct_charges:
            sum_charge = snn.scatter_add(charges, idx_m, dim_size=maxm)
            total_charge = inputs.get(
                properties.total_charge, torch.zeros_like(sum_charge)
            )[:, None]
            charge_correction = (total_charge - sum_charge) / natoms.unsqueeze(-1)
            charges = charges + charge_correction[idx_m]

        # Store partial charges if required
        if self.return_charges:
            inputs[self.charges_key] = charges

        # Compute dipole moment from charges and positions + atomic dipoles (if used)
        y = positions * charges + atomic_dipoles
        y = snn.scatter_add(y, idx_m, dim_size=maxm)

        # Predict magnitude instead of vector if required
        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=False)

        inputs[self.dipole_key] = y
        return inputs


class Polarizability(nn.Module):
    """
    Predicts polarizability tensor using tensor rank factorization.
    Requires an equivariant representation (e.g., PaiNN) providing scalar and vector features.

    References:
    .. [#painn1a] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
            self,
            n_in: int,
            n_hidden: Optional[Union[int, Sequence[int]]] = None,
            n_layers: int = 2,
            activation: Callable = F.silu,
            polarizability_key: str = properties.polarizability,
    ):
        super().__init__()
        self.polarizability_key = polarizability_key
        self.model_outputs = [polarizability_key]

        # Build equivariant MLP for polarizability prediction
        self.outnet = spk.nn.build_gated_equivariant_mlp(
            n_in=n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
            sactivation=activation,
        )

        self.requires_dr = False
        self.requires_stress = False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = inputs[properties.R]
        l0 = inputs["scalar_representation"]
        l1 = inputs["vector_representation"]
        dim = l1.shape[-2]

        # Process features through equivariant network
        l0, l1 = self.outnet((l0, l1))

        # Isotropic contribution (diagonal terms)
        alpha = l0[..., 0:1].expand(-1, -1, dim)  # Shape: (n_atoms, 1, dim) -> (n_atoms, dim)
        alpha = torch.diag_embed(alpha)  # Shape: (n_atoms, dim, dim)

        # Anisotropic contribution (rank-1 tensor product)
        mur = l1[..., None, 0] * positions[..., None, :]  # Shape: (n_atoms, dim, dim)
        alpha_c = mur + mur.transpose(-2, -1)  # Symmetrize
        alpha = alpha + alpha_c

        # Aggregate over atoms
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        alpha = snn.scatter_add(alpha, idx_m, dim_size=maxm)

        inputs[self.polarizability_key] = alpha
        return inputs