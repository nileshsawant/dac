"""
Module 2: Data-Driven Design Surrogate (GNN)
============================================

This module implements an Equivariant Graph Neural Network (eGNN) for predicting
sorbent performance across ambient climates using a "differential descriptor" approach.

Key Features:
- Equivariant Graph Neural Network architecture
- Differential descriptor calculation (product - reactant)
- Multi-target prediction: CO2 working capacity, H2O penalty, regeneration favorability
- Integration with RDKit and PyTorch Geometric
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SorbentDescriptors:
    """Container for sorbent descriptors."""
    amine_density: float  # Local amine site density (sites/nm²)
    hydrophobicity: float  # Contact angle proxy (degrees)
    pore_size: float  # Confinement proxy (Å)
    backbone_flexibility: float  # Torsional barrier proxy (kcal/mol)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.amine_density,
            self.hydrophobicity,
            self.pore_size,
            self.backbone_flexibility
        ])


@dataclass
class PerformanceTargets:
    """Container for performance prediction targets."""
    co2_capacity: float  # CO2 working capacity (mol/kg)
    h2o_penalty: float  # H2O co-adsorption penalty (%)
    regen_temperature: float  # Regeneration temperature threshold (°C)
    regen_vacuum: Optional[float] = None  # Vacuum threshold (mbar)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        targets = [self.co2_capacity, self.h2o_penalty, self.regen_temperature]
        if self.regen_vacuum is not None:
            targets.append(self.regen_vacuum)
        return np.array(targets)


class DifferentialDescriptor:
    """
    Calculate differential descriptors between reactant and product molecules.
    
    The differential descriptor approach predicts binding enthalpy by computing
    the difference in molecular descriptors between product and reactant states.
    """
    
    def __init__(self):
        """Initialize differential descriptor calculator."""
        self.descriptor_names = []
        
    def mol_to_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate molecular descriptors for a single molecule.
        
        Parameters:
        -----------
        mol : Chem.Mol
            RDKit molecule object
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of descriptor names and values
        """
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),
            'BalabanJ': Descriptors.BalabanJ(mol),
        }
        
        return descriptors
    
    def calculate_differential(self,
                             product_mol: Chem.Mol,
                             reactant_mols: List[Chem.Mol]) -> np.ndarray:
        """
        Calculate differential descriptors.
        
        Δdescriptor = descriptor(product) - Σ descriptor(reactants)
        
        Parameters:
        -----------
        product_mol : Chem.Mol
            Product molecule
        reactant_mols : List[Chem.Mol]
            List of reactant molecules
            
        Returns:
        --------
        np.ndarray
            Array of differential descriptors
        """
        product_desc = self.mol_to_descriptors(product_mol)
        
        # Sum reactant descriptors
        reactant_desc_sum = {}
        for mol in reactant_mols:
            desc = self.mol_to_descriptors(mol)
            for key, val in desc.items():
                reactant_desc_sum[key] = reactant_desc_sum.get(key, 0.0) + val
        
        # Calculate differences
        differential = {}
        for key in product_desc.keys():
            differential[key] = product_desc[key] - reactant_desc_sum[key]
        
        self.descriptor_names = list(differential.keys())
        return np.array(list(differential.values()))
    
    def smiles_to_differential(self,
                              product_smiles: str,
                              reactant_smiles: List[str]) -> np.ndarray:
        """
        Calculate differential descriptors from SMILES strings.
        
        Parameters:
        -----------
        product_smiles : str
            SMILES string for product
        reactant_smiles : List[str]
            List of SMILES strings for reactants
            
        Returns:
        --------
        np.ndarray
            Array of differential descriptors
        """
        product_mol = Chem.MolFromSmiles(product_smiles)
        reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
        
        return self.calculate_differential(product_mol, reactant_mols)


class EGNNLayer(MessagePassing):
    """
    Equivariant Graph Neural Network layer.
    
    Maintains equivariance to rotations and translations in 3D space.
    """
    
    def __init__(self, 
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 128):
        """
        Initialize EGNN layer.
        
        Parameters:
        -----------
        node_features : int
            Number of node features
        edge_features : int
            Number of edge features
        hidden_dim : int
            Hidden dimension size
        """
        super().__init__(aggr='add')
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_features + edge_features + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(node_features + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_features)
        )
        
        # Coordinate update MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(self, 
                x: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [N, node_features]
        pos : torch.Tensor
            Node positions [N, 3]
        edge_index : torch.Tensor
            Edge connectivity [2, E]
        edge_attr : torch.Tensor
            Edge features [E, edge_features]
            
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Updated node features and positions
        """
        # Compute pairwise distances
        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        dist = torch.norm(rel_pos, dim=1, keepdim=True)
        
        # Message passing
        x_new = self.propagate(edge_index, x=x, pos=pos, 
                              edge_attr=edge_attr, dist=dist)
        
        # Update node features
        x = x + self.node_mlp(torch.cat([x, x_new], dim=-1))
        
        # Update coordinates (equivariant)
        coord_weights = self.coord_mlp(x_new)
        pos = pos + (rel_pos / (dist + 1e-8)) * coord_weights[row]
        
        return x, pos
    
    def message(self, 
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                dist: torch.Tensor) -> torch.Tensor:
        """Construct messages."""
        msg_input = torch.cat([x_i, x_j, edge_attr, dist], dim=-1)
        return self.message_mlp(msg_input)


class GNNSurrogate(nn.Module):
    """
    Equivariant Graph Neural Network for sorbent performance prediction.
    
    Architecture based on UMA (Universal Molecular Approximator) design.
    """
    
    def __init__(self,
                 node_features: int = 64,
                 edge_features: int = 16,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_targets: int = 3,
                 use_differential: bool = True):
        """
        Initialize GNN surrogate model.
        
        Parameters:
        -----------
        node_features : int
            Number of node features
        edge_features : int
            Number of edge features
        hidden_dim : int
            Hidden dimension size
        num_layers : int
            Number of EGNN layers
        num_targets : int
            Number of prediction targets (default: 3)
            - CO2 capacity
            - H2O penalty
            - Regeneration temperature
        use_differential : bool
            Whether to use differential descriptors
        """
        super().__init__()
        
        self.num_targets = num_targets
        self.use_differential = use_differential
        
        # Node embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Edge embedding
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        
        # Differential descriptor network (if used)
        if use_differential:
            self.diff_encoder = nn.Sequential(
                nn.Linear(11, hidden_dim // 4),  # 11 differential descriptors
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim // 4)
            )
        
    def forward(self, 
                data: Data,
                diff_descriptors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        data : Data
            PyTorch Geometric Data object containing:
            - x: node features
            - pos: 3D positions
            - edge_index: connectivity
            - edge_attr: edge features
            - batch: batch assignment
        diff_descriptors : torch.Tensor, optional
            Differential descriptors [batch_size, 11]
            
        Returns:
        --------
        torch.Tensor
            Predictions [batch_size, num_targets]
        """
        x = self.node_embedding(data.x)
        pos = data.pos
        edge_attr = self.edge_embedding(data.edge_attr)
        
        # Apply EGNN layers
        for layer in self.egnn_layers:
            x, pos = layer(x, pos, data.edge_index, edge_attr)
        
        # Global pooling
        graph_embedding = global_mean_pool(x, data.batch)
        
        # Combine with differential descriptors if provided
        if self.use_differential and diff_descriptors is not None:
            diff_embedding = self.diff_encoder(diff_descriptors)
            graph_embedding = torch.cat([graph_embedding, diff_embedding], dim=-1)
            
            # Adjust predictor input size
            if not hasattr(self, '_adjusted_predictor'):
                input_dim = graph_embedding.shape[1]
                self.predictor[0] = nn.Linear(input_dim, self.predictor[0].out_features)
                self._adjusted_predictor = True
        
        # Predict performance metrics
        predictions = self.predictor(graph_embedding)
        
        return predictions
    
    def train_model(self,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   num_epochs: int = 100,
                   learning_rate: float = 1e-3,
                   device: str = 'cpu') -> Dict[str, List[float]]:
        """
        Train the GNN model.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        num_epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate
        device : str
            Device to train on ('cpu' or 'cuda')
            
        Returns:
        --------
        Dict[str, List[float]]
            Training history (train_loss, val_loss, val_mae)
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                pred = self(batch, batch.diff_desc if hasattr(batch, 'diff_desc') else None)
                loss = F.mse_loss(pred, batch.y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.eval()
            val_loss = 0.0
            val_mae = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = self(batch, batch.diff_desc if hasattr(batch, 'diff_desc') else None)
                    
                    val_loss += F.mse_loss(pred, batch.y).item()
                    val_mae += F.l1_loss(pred, batch.y).item()
            
            val_loss /= len(val_loader)
            val_mae /= len(val_loader)
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val MAE: {val_mae:.4f}")
        
        return history
    
    def predict(self, 
                data: Data,
                diff_descriptors: Optional[torch.Tensor] = None,
                device: str = 'cpu') -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        data : Data
            Input molecular graph
        diff_descriptors : torch.Tensor, optional
            Differential descriptors
        device : str
            Device to use
            
        Returns:
        --------
        np.ndarray
            Predictions [num_targets]
        """
        self.eval()
        self.to(device)
        data = data.to(device)
        
        if diff_descriptors is not None:
            diff_descriptors = diff_descriptors.to(device)
        
        with torch.no_grad():
            pred = self(data, diff_descriptors)
        
        return pred.cpu().numpy()


def mol_to_graph(mol: Chem.Mol, 
                conformer_id: int = -1) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric Data object.
    
    Parameters:
    -----------
    mol : Chem.Mol
        RDKit molecule
    conformer_id : int
        Conformer ID (default: -1 for most recent)
        
    Returns:
    --------
    Data
        PyTorch Geometric Data object
    """
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Get 3D coordinates
    if mol.GetNumConformers() > 0:
        conformer = mol.GetConformer(conformer_id)
        pos = conformer.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
    else:
        # Generate 3D coordinates if not present
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        conformer = mol.GetConformer()
        pos = conformer.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
    
    # Get edge information
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_features = [
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ]
        
        # Add both directions
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bond_features, bond_features])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
