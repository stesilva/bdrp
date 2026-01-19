# RGCN Baseline for OntoOmicsKG

This directory contains the standard RGCN (Relational Graph Convolutional Network) implementation for link prediction on the OntoOmicsKG dataset.

## Files

- `main.py` - Main training script
- `models.py` - RGCN model definition
- `utils.py` - Data loading and utility functions (adapted for OntoOmicsKG format)
- `run_rgcn_ontoomics_ruche.sh` - SLURM script to run on Ruche cluster

## Usage

### Local Training

```bash
cd OntoOmicsKG/rgcn
python main.py --gpu 0 --data-path ..
```

### Running on Ruche

```bash
cd OntoOmicsKG/rgcn
sbatch run_rgcn_ontoomics_ruche.sh
```

## Data Format

The script expects OntoOmicsKG data files in the parent directory (`../`):
- `edges.filtered.tsv.entities.tsv` - Entity ID mappings
- `edges.filtered.tsv.relations.tsv` - Relation ID mappings
- `edges_train.tsv` - Training triplets
- `edges_val.tsv` - Validation triplets
- `edges_test.tsv` - Test triplets

## Results

The trained model is saved as `best_mrr_model.pth` when validation MRR improves.

## Hyperparameters

Default settings:
- Epochs: 10,000
- Evaluation every: 500 epochs
- Batch size: 30,000
- Learning rate: 0.01
- Dropout: 0.2
- Number of bases: 4

