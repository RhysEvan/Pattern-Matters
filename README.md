# Projection Pattern Optimization
## Paper title: Structured light pattern optimization for machine learning-based single-shot profilometry

This repository contains tools for generating simulated datasets and training machine learning models. The structure is organized into four main folders:

1. **Data Generation**
   - Gaussian Surface
   - Perlin Surface
2. **Network Training**
   - SGD Optimizer
   - ADAM Optimizer

Each folder has a specific purpose and contains the necessary files to complete its respective tasks.

---

## Folder Structure

### 1. Data Generation

#### **Gaussian Surface**
This folder provides tools to generate a simulated Gaussian surface dataset. The user should follow the file order starting from **`a`** to **`b`** to create a complete dataset for training.

- **Key Files:**
  - `A`: Generates the **`label`** on which projection patterns get applied.
  - `B`: Projects the period patterns onto the surface resulting in the **`input`** data for the neural network training.

#### **Perlin Surface**
This folder contains tools for generating a simulated Perlin surface dataset. Users should follow the files in order, from **`A`** to **`C`**, to generate the primary dataset. Additional files are available for background work or testing, but they are not required for the main functionality.

- **Key Files:**
  - `A`: Generates the **`label`** on which projection patterns get applied.
  - `B`: Projects the period patterns onto the surface resulting in the **`input`** data for the neural network training.
  - `C`: Projects the period patterns onto the surface resulting in the **`input`** data for specifically the Denominator/Nominator neural network training.

---

### 2. Network Training

Each network folder corresponds to a different optimization algorithm and includes code for training a machine learning model.

#### **SGD Optimizer**
This folder contains the code for training a model using the Stochastic Gradient Descent (SGD) optimizer.

- **Subfolder:**
  - `model`: Contains all the model-specific code.

- **Subfiles:**
  - `training_...`: Contains the code to train the model associated with this optimizer.

#### **ADAM Optimizer**
This folder contains the code for training a model using the ADAM optimizer.

- **Subfolders:**
  - `model`: Contains all the model-specific code.

- **Subfiles:**
  - `training_...`: Contains the code to train the model associated with this optimizer.

---

## Usage Instructions

### Generating Data
1. **Gaussian Surface:** Navigate to the `Gaussian Surface` folder and execute the files in order (`A` to `B`).
2. **Perlin Surface:** Navigate to the `Perlin Surface` folder and execute the files in order (`A` to `C`).
   - Note: Additional files are available for background work or optional tests but are not necessary for the primary workflow.

### Training Models
1. Select the network folder corresponding to your desired optimizer (SGD or ADAM).
2. Navigate to the `training_...` file and assure that hyperparameters are correctly set up at the bottom of the script.
3. The `model` folder contains all the model code if you need to customize or inspect it.

---

## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.