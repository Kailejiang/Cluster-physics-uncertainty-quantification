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
        self.nig_params_cache = None 
        self.uncertainty_cache = None 
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

        nig_params = self.nig_head(y)
        self.nig_params_cache = nig_params 

        mu, v, alpha, beta = torch.split(nig_params, self.n_out, dim=-1)
        v = F.softplus(v) + 1e-6
        alpha = F.softplus(alpha) + 1.0 + 1e-6
        beta = F.softplus(beta) + 1e-6
        epistemic_uncertainty = beta / (alpha - 1) / v
        aleatoric_uncertainty = beta / (alpha - 1)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        self.uncertainty_cache = {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
            "total": total_uncertainty
        }

        mu, _, _, _ = torch.split(nig_params, self.n_out, dim=-1)
        inputs[self.output_key] = mu.squeeze(-1) if self.n_out == 1 else mu

        if not self.training:
            with torch.no_grad():

                atom_feat = inputs["scalar_representation"]
                idx_m = inputs[properties.idx_m]
                maxm = int(idx_m[-1]) + 1
                mol_feat = snn.scatter_add(atom_feat, idx_m, dim_size=maxm)
                if self.aggregation_mode == "avg":
                    mol_feat = mol_feat / inputs[properties.n_atoms]
                energy = inputs[self.output_key]

                if self.uncertainty_cache is not None:
                    uncertainty = self.uncertainty_cache["total"]
                else:
                    uncertainty = torch.zeros_like(energy)

                self.tsne_features.append(mol_feat.detach().to("cpu"))
                self.tsne_energies.append(energy.detach().to("cpu"))
                self.tsne_uncertainties.append(uncertainty.detach().to("cpu"))

        return inputs

    def get_cached_nig_params(self):
        if self.nig_params_cache is None:
            raise RuntimeError("请先调用forward方法生成NIG参数后，再获取缓存！")
        return self.nig_params_cache

    def get_cached_uncertainty(self):
        if self.uncertainty_cache is None:
            raise RuntimeError("请先调用forward方法生成不确定性后，再获取缓存！")
        return self.uncertainty_cache

    def export_tsne_csv(self, save_path="tsne_results.csv"):

        X = torch.cat([t.cpu() for t in self.tsne_features], dim=0).numpy()
        y = torch.cat([t.cpu() for t in self.tsne_energies], dim=0).numpy()
        u = torch.cat([t.cpu() for t in self.tsne_uncertainties], dim=0).numpy()

        print(f"t-SNE input dimension: {X.shape}")

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

        print(f"t-SNE saved: {save_path}")

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
        self.mc_samples = mc_samples  
        self.mc_predictions_cache = None  
        self.uncertainty_cache = None  

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

        if self.training or (not self.training and self.mc_samples > 1):
            mc_predictions = []
            self.outnet.train()
            for _ in range(self.mc_samples):
                y_mc = self.outnet(inputs["scalar_representation"])
                if self.aggregation_mode is not None:
                    y_mc = snn.scatter_add(y_mc, idx_m, dim_size=maxm)
                    y_mc = torch.squeeze(y_mc, -1)
                    if self.aggregation_mode == "avg":
                        y_mc = y_mc / inputs[properties.n_atoms]
                mc_predictions.append(y_mc.unsqueeze(0))
            mc_predictions = torch.cat(mc_predictions, dim=0)
            self.mc_predictions_cache = mc_predictions

            pred_mean = torch.mean(mc_predictions, dim=0)  
            pred_var = torch.var(mc_predictions, dim=0)    
            self.uncertainty_cache = {
                "mean": pred_mean,   
                "var": pred_var,     
                "std": torch.sqrt(pred_var + 1e-6)  
            }

        if not self.training:
            with torch.no_grad():

                atom_feat = inputs["scalar_representation"]
                idx_m = inputs[properties.idx_m]
                maxm = int(idx_m[-1]) + 1
                mol_feat = snn.scatter_add(atom_feat, idx_m, dim_size=maxm)

                if self.aggregation_mode == "avg":
                    mol_feat = mol_feat / inputs[properties.n_atoms]

                if self.uncertainty_cache is not None:
                    energy = self.uncertainty_cache["mean"]
                    uncertainty = self.uncertainty_cache["std"]
                else:
                    energy = inputs[self.output_key]
                    uncertainty = torch.zeros_like(energy)

                self.tsne_features.append(mol_feat.detach().to("cpu"))
                self.tsne_energies.append(energy.detach().to("cpu"))
                self.tsne_uncertainties.append(uncertainty.detach().to("cpu"))

        return inputs

    def export_tsne_csv(self, save_path="tsne_results.csv"):

        X = torch.cat([t.cpu() for t in self.tsne_features], dim=0).numpy()
        y = torch.cat([t.cpu() for t in self.tsne_energies], dim=0).numpy()
        u = torch.cat([t.cpu() for t in self.tsne_uncertainties], dim=0).numpy()

        print(f"t-SNE input dimension: {X.shape}")

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

        print(f"t-SNE saved: {save_path}")

    def reset_tsne_cache(self):
        self.tsne_features = []
        self.tsne_energies = []
        self.tsne_uncertainties = []

    def get_cached_mc_predictions(self):
        if self.mc_predictions_cache is None:
            raise RuntimeError("请先调用forward方法完成蒙特卡洛采样后，再获取缓存！")
        return self.mc_predictions_cache

    def get_cached_uncertainty(self):
        if self.uncertainty_cache is None:
            raise RuntimeError("请先调用forward方法完成蒙特卡洛采样后，再获取不确定性缓存！")
        return self.uncertainty_cache


class BayesianNN(nn.Module):

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
        mc_samples: int = 50, 
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
        self.kl_div_cache = None 
        self.mc_predictions_cache = None  
        self.uncertainty_cache = None  
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
        total_kl = 0.0
        for module in self.outnet.modules():
            if isinstance(module, snn.BayesianDense):
                total_kl += module.kl_divergence()
        return total_kl

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.kl_div_cache = None
        self.mc_predictions_cache = None
        self.uncertainty_cache = None

        self.kl_div_cache = self._compute_total_kl_div()

        y = self.outnet(inputs["scalar_representation"])

        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[properties.n_atoms]

        inputs[self.output_key] = y

        if self.mc_samples > 1:
            mc_predictions = []
            for _ in range(self.mc_samples):
                y_mc = self.outnet(inputs["scalar_representation"])

                if self.aggregation_mode is not None:
                    y_mc = snn.scatter_add(y_mc, idx_m, dim_size=maxm)
                    y_mc = torch.squeeze(y_mc, -1)
                    if self.aggregation_mode == "avg":
                        y_mc = y_mc / inputs[properties.n_atoms]
                mc_predictions.append(y_mc.unsqueeze(0))

            mc_predictions = torch.cat(mc_predictions, dim=0)
            self.mc_predictions_cache = mc_predictions


            pred_mean = torch.mean(mc_predictions, dim=0)
            pred_var = torch.var(mc_predictions, dim=0)
            self.uncertainty_cache = {
                "mean": pred_mean,
                "var": pred_var,
                "std": torch.sqrt(pred_var + 1e-6) 
            }

        if not self.training:
            with torch.no_grad():

                atom_feat = inputs["scalar_representation"]

                idx_m = inputs[properties.idx_m]
                maxm = int(idx_m[-1]) + 1

                mol_feat = snn.scatter_add(atom_feat, idx_m, dim_size=maxm)

                if self.aggregation_mode == "avg":
                    mol_feat = mol_feat / inputs[properties.n_atoms]

                energy = inputs[self.output_key]

                if self.uncertainty_cache is not None:
                    uncertainty = self.uncertainty_cache["std"]
                else:
                    uncertainty = torch.zeros_like(energy)

                self.tsne_features.append(mol_feat.detach().cpu())
                self.tsne_energies.append(energy.detach().cpu())
                self.tsne_uncertainties.append(uncertainty.detach().cpu())

        return inputs

    def export_tsne_csv(self, save_path="tsne_results.csv"):


        X = torch.cat([t.cpu() for t in self.tsne_features], dim=0).numpy()
        y = torch.cat([t.cpu() for t in self.tsne_energies], dim=0).numpy()
        u = torch.cat([t.cpu() for t in self.tsne_uncertainties], dim=0).numpy()

        print(f"t-SNE input dimension: {X.shape}")

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

        print(f"t-SNE saved: {save_path}")

    def reset_tsne_cache(self):
        self.tsne_features = []
        self.tsne_energies = []
        self.tsne_uncertainties = []

    def get_cached_kl_div(self) -> torch.Tensor:
        if self.kl_div_cache is None:
            raise RuntimeError("请先调用forward方法后，再获取KL散度缓存！")
        return self.kl_div_cache

    def get_cached_mc_predictions(self) -> torch.Tensor:
        if self.mc_predictions_cache is None:
            raise RuntimeError("请先调用forward方法并设置mc_samples>1后，再获取采样缓存！")
        return self.mc_predictions_cache

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
