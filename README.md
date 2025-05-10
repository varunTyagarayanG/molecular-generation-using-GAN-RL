# Molecular Generation with GANs and Reinforcement Learning

This project focuses on generating novel molecular structures using Generative Adversarial Networks (GANs) and reinforcement learning techniques. The goal is to create valid, diverse, and optimized molecules for various applications in drug discovery and material science.

## Project Structure

- **`layers.py`**: Contains custom neural network layers used in the GAN architecture.
- **`model_1.5_Reward.py`, `model_2.0_Reward.py`, `model_2.5_Reward.py`**: Different versions of the GAN model with varying reward mechanisms.
- **`tokenizer.py`**: Handles the tokenization of SMILES strings for molecular representation.
- **`notebook1.5Reward.ipynb`, `notebook2.0Reward.ipynb`, `notebook2.5Reward.ipynb`**: Jupyter notebooks for training and evaluating the GAN models.
- **`qm9.csv`**: Dataset containing molecular SMILES strings.
- **`generated_smiles.txt`**: Output file containing generated SMILES strings.
- **`new_molecules.txt`, `old_molecules.txt`**: Files categorizing generated molecules as novel or existing.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch
- RDKit

Install the required dependencies using:
```bash
pip install -r requirement.txt
```

### Running the Project

1. Preprocess the dataset:
   ```bash
   python tokenizer.py
   ```

2. Train the model using one of the provided notebooks:
   - `notebook1.5Reward.ipynb`
   - `notebook2.0Reward.ipynb`
   - `notebook2.5Reward.ipynb`

3. Evaluate the generated molecules and compute metrics such as validity, novelty, diversity, and QED.

## Results

The generated molecules are saved in `generated_smiles.txt`. Metrics such as validity, novelty, and diversity are computed during training and can be visualized using the provided notebooks.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- RDKit for molecular processing
- PyTorch for deep learning
- QM9 dataset for molecular data