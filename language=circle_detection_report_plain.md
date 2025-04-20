This experience aligns with the general pattern that Adam often performs better "out of the box" for deep learning tasks, while SGD may require more careful tuning and stronger regularization to achieve comparable results.

### Architectural Enhancements
| Action                             | Outcome                  |
|------------------------------------|---------------------------|
| Added `conv3` layer                | Improved depth & features |
| Used `BatchNorm` after each conv   | Better convergence        |
| Switched to Global Avg Pooling     | Lower param count         |
| Removed Dropout                    | Simpler, less regularized |
| Removed flatten                    | Prevented large FC layer  |

While many optimization attempts were not fully successful due to time limitations, the complexity of CNN architecture design means these methods are still worth exploring with different combinations. The interaction between regularization techniques, optimization algorithms, and network architecture creates a vast design space that couldn't be exhaustively searched within project constraints. Future work would benefit from systematic experimentation with various combinations of these techniques, potentially yielding further improvements to the model's performance.

---

###  Loss Function Experiments 