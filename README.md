# Trellis BMA: coded trace reconstruction on IDS channels for DNA storage

This repo contains an implementation of Trellis BMA algorithm from [https://arxiv.org/abs/2107.06440](https://arxiv.org/abs/2107.06440) by Sundara Rajan Srinivasavaradhan, Sivakanth Gopi, Henry D. Pfister and Sergey Yekhanin.

Abstract: *Sequencing a DNA strand, as part of the read process in DNA storage, produces multiple noisy copies which can be combined to produce better estimates of the original strand; this is called trace reconstruction. One can reduce the error rate further by introducing redundancy in the write sequence and this is called coded trace reconstruction. In this paper, we model the DNA storage channel as an insertion-deletion-substitution (IDS) channel and design both encoding schemes and low-complexity decoding algorithms for coded trace reconstruction.
We introduce Trellis BMA, a new reconstruction algorithm whose complexity is linear in the number of traces, and compare its performance to previous algorithms. Our results show that it reduces the error rate on both simulated and experimental data. The performance comparisons in this paper are based on a new dataset of traces that will be publicly released with the paper. Our hope is that this dataset will enable research progress by allowing objective comparisons between candidate algorithms.*


## Quickstart instructions

Please start with "Documentation_IDS_CC.ipynb" to understand usage and how to call Trellis BMA (and other algorithms). The documentation contains instructions for:
- building a finite-state machine corresponding to a convolutional code (substitute this with similar code for your usecase)
- building the IDS trellis on top of the FSM
- running trellis BMA on the IDS trellis
- visualization tools

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
