import os
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torchmetrics
import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn
from sklearn.metrics import pairwise_distances
from schnetpack.utils import load_model
from ase import Atoms
from schnetpack.data import ASEAtomsData


# =========================
# Load config
# =========================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

dataset_dir = cfg["paths"]["dataset_dir"]
output_dir = cfg["paths"]["output_dir"]
model_dir = cfg["paths"]["model_dir"]

batch_size = cfg["data"]["batch_size"]
cutoff = cfg["model"]["cutoff"]


# =========================
# Paths
# =========================
save_path = f"{output_dir}/evidentialNN"
save_str_dir = f"{output_dir}/{cfg['files']['structure_dir']}"
db_path = f"{output_dir}/{cfg['files']['db_out']}"
csv_path = f"{output_dir}/{cfg['files']['csv']}"
tsne_path = f"{output_dir}/{cfg['files']['tsne']}"

os.makedirs(save_str_dir, exist_ok=True)


# =========================
# Data
# =========================
train_data = spk.data.AtomsDataModule(
    f"{dataset_dir}/{cfg['data']['train_db']}",
    batch_size=batch_size,
    distance_unit="Ang",
    property_units={"energy_U0": "eV"},
    num_train=0.9,
    num_val=0.1,
    split_file=cfg["data"]["split_train"],
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        trn.RemoveOffsets("energy_U0", remove_mean=True),
        trn.CastTo32(),
    ],
)

predict_data = spk.data.AtomsDataModule(
    datapath=f"{dataset_dir}/{cfg['data']['predict_db']}",
    batch_size=batch_size,
    distance_unit="Ang",
    property_units={"energy_U0": "eV"},
    num_train=0.9,
    num_val=0.1,
    split_file=cfg["data"]["split_pred"],
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        trn.RemoveOffsets("energy_U0", remove_mean=True),
        trn.CastTo32(),
    ],
)


train_data.prepare_data()
predict_data.prepare_data()


# =========================
# Train
# =========================
train_data.setup()

pairwise = spk.atomistic.PairwiseDistances()
rbf = spk.nn.GaussianRBF(n_rbf=cfg["model"]["n_rbf"], cutoff=cutoff)

schnet = spk.representation.SchNet(
    n_atom_basis=cfg["model"]["n_atom_basis"],
    n_interactions=3,
    radial_basis=rbf,
    cutoff_fn=spk.nn.CosineCutoff(cutoff),
)

pred = spk.atomistic.EvidentialNN(
    n_in=cfg["model"]["n_atom_basis"],
    output_key="energy_U0",
)

model = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise],
    output_modules=pred,
    postprocessors=[
        trn.CastTo64(),
        trn.AddOffsets("energy_U0", add_mean=True),
    ],
)

loss_fn = spk.task.EvidentialLoss(reg_coeff=0.1)

output = spk.task.ModelOutput_ENN(
    name="energy_U0",
    loss_fn=loss_fn,
    loss_weight=1.0,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError(),
        "MSE": torchmetrics.MeanSquaredError(),
    },
)

task = spk.task.AtomisticTask_ENN(
    model=model,
    outputs=[output],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": cfg["model"]["lr"]},
)


trainer = pl.Trainer(
    max_epochs=cfg["model"]["max_epochs"],
    default_root_dir=output_dir,
    logger=pl.loggers.CSVLogger(output_dir, name="enn_run"),
)


trainer.fit(task, datamodule=train_data)


# =========================
# Prediction
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predict_data.setup()

model = load_model(
    f"{model_dir}/{cfg['files']['checkpoint']}",
    device=device,
)

model.eval()
model.output_modules.reset_tsne_cache()


energies, uncertainties, structures = [], [], []

for batch in predict_data.train_dataloader():

    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    with torch.no_grad():
        out = model(batch)

    energies.append(out["energy_U0"].cpu().numpy())

    unc = model.output_modules.get_cached_uncertainty()["total"].cpu().numpy()
    uncertainties.append(unc)

    if "_positions" in batch:
        n_atoms = batch["_n_atoms"][0].cpu().numpy()
        pos = batch["_positions"].cpu().numpy().reshape(batch_size, n_atoms, 3)
        structures.append(pos)


energies = np.concatenate(energies)
uncertainties = np.concatenate(uncertainties)
structures = np.concatenate(structures)


model.output_modules.export_tsne_csv(tsne_path)


# =========================
# Save CSV
# =========================
pd.DataFrame({
    "Energy": energies,
    "Uncertainty": uncertainties
}).to_csv(csv_path, index=False)


# =========================
# Representative selection
# =========================
idx_low = np.argmin(energies)
idx_high = np.argmax(energies)
idx_u_high = np.argmax(uncertainties)
idx_u_low = np.argmin(uncertainties)


def save_xyz(structure, energy, unc, name):
    with open(os.path.join(save_str_dir, name), "w") as f:
        f.write(f"{len(structure)}\n")
        f.write(f"Energy: {energy:.6f}, Uncertainty: {unc:.6f}\n")
        for c in structure:
            f.write(f"X {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


save_xyz(structures[idx_low], energies[idx_low], uncertainties[idx_low], "low_energy.xyz")
save_xyz(structures[idx_high], energies[idx_high], uncertainties[idx_high], "high_energy.xyz")
save_xyz(structures[idx_u_high], energies[idx_u_high], uncertainties[idx_u_high], "high_uncertainty.xyz")
save_xyz(structures[idx_u_low], energies[idx_u_low], uncertainties[idx_u_low], "low_uncertainty.xyz")


# =========================
# Active Learning selection
# =========================
threshold = np.percentile(uncertainties, 90)
high_idx = np.where(uncertainties > threshold)[0]

high_struct = structures[high_idx]
high_energy = energies[high_idx]

dist = pairwise_distances(high_struct.reshape(len(high_struct), -1))
mean_dist = np.mean(dist, axis=1)

sel = np.where(mean_dist > np.percentile(mean_dist, 80))[0]

selected_struct = high_struct[sel]
selected_energy = high_energy[sel]

