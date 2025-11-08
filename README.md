# AlphaLab
## Introduction of PL_Alpha_Hill

* **What it is.** A per-layer *power-law exponent* fitted to the heavy-tailed part of a layer’s eigenvalue spectrum (the ESD) of its weight correlation matrix. Practically, you compute the ESD for each layer and fit a power law on its tail; the fitted exponent is called **α** (often reported as **PL_Alpha_Hill**).  
* **Why it matters.** In Heavy-Tailed Self-Regularization (HT-SR), layers whose ESDs are *more heavy-tailed* tend to be *better trained*; α captures this: **smaller α ⇒ heavier tail ⇒ higher training quality; larger α ⇒ lighter tail ⇒ undertrained**.  
* **How it’s used.**

  * **Pruning (AlphaPruning).** Use α to allocate sparsity: keep more parameters in layers with **lower α** (more heavy-tailed, better trained), prune more where **α is higher**.  
  * **Learning-rate scheduling (TempBalance).** Map each layer’s α to a layer-wise learning rate: smaller rates for heavy-tailed layers, larger rates for light-tailed layers, to balance training.  
  * **Weight-decay scheduling (AlphaDecay).** Assign *lower* decay to modules with **lower α** (more heavy-tailed) and *higher* decay where **α** is larger, improving module-wise balance. 
* **Context.** HT-SR tools start by analyzing each layer’s ESD; fitting a power law provides α as a compact shape metric of the ESD that correlates with layer quality.  
 


## Usage
```
python main.py mistralai/Mixtral-8x7B-v0.1 --output-csv data/mixtral_alpha_base.csv
```