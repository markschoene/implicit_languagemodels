# Implicit Languagemodels
This repository contains the code for our publication, [**"Implicit Languagemodels are RNN: Balancing Parallelization and Expressivity"**](https://arxiv.org/abs/2502.07827).

### ICML 2025 Spotlight
🎉🚀 We are delighted to announce that our paper was spotlighted at ICML 2025! 🚀🎉

### Description

- The `implicit_llm` folder is importable when this repository is installed as a package. The core implementation of implicit models is defined in `implicit_llm/implicit.py`.
- Model architectures are defined in the `hf_models` directory.
- Examples for downstream evaluations can be found in `examples/downstream_evaluation.py`, while an example of improved state-tracking (S5) is provided in `examples/state_tracking.py`.
- State-tracking datasets for Figures 1 Top Left/Top Right and Figure 3 in the paper are available in the `state_tracking` folder.
- An example how to calculate the implicit Jacobian as discussed in Theorem 1 (see also A2 and Figure 7) is provided in the implicit_jacobian notebook.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
