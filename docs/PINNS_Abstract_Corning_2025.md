# Physics-Informed Neural Networks Abstract
## Corning Future Innovator Program 2025
**Advanced Temperature Interpolation for Ceramic Tile Manufacturing**

---

## Slide 1: Literature Review: Physics-Informed Neural Networks for Thermal Interpolation

• Traditional interpolation methods (bilinear, RBF, kriging) lack physics constraints for ceramic firing

• Standard neural networks require extensive training data and may violate physical laws

• PINNs combine deep learning flexibility with governing equation enforcement (Raissi et al., 2019)

• Limited application to sparse sensor interpolation in high-temperature manufacturing processes

• Gap: No PINNs framework for real-time ceramic tile temperature prediction with sensor failures

• Heat transfer physics essential: convection, radiation, chemical reaction heat generation

• Need for uncertainty quantification and robustness in industrial thermal monitoring

---

## Slide 2: Novelty: Physics-Constrained Neural Temperature Interpolation

• Novel PINN architecture enforcing heat equation: ρcp(∂T/∂t) = ∇·(k∇T) + GE

• Multi-component loss function: L = L_data + λ₁L_physics + λ₂L_boundary + λ₃L_initial

• Adaptive weight scheduling for physics vs. data terms during training

• Integrated chemical reaction modeling for exothermic/endothermic clay transformations

• Sensor failure detection through residual analysis and physics violation metrics

• Uncertainty quantification via ensemble PINNs and Monte Carlo dropout

• Real-time inference capability with pre-trained physics-aware neural interpolator

---

## Slide 3: Execution Plan: PINN Development and Validation Strategy

• Phase 1: Implement baseline PINN with 2D heat equation for cylindrical geometry

• Phase 2: Integrate boundary conditions (convection/radiation) and material properties

• Phase 3: Add chemical reaction terms and validate against Data A (15→120 interpolation)

• Phase 4: Develop sensor failure detection using physics residuals and ensemble uncertainty

• Phase 5: Optimize hyperparameters (λ weights, network architecture, training schedule)

• Phase 6: Final validation on Data B with comprehensive error analysis and robustness testing

• Expected outcome: <5°C RMSE with physics-consistent predictions and sensor fault tolerance

---

## Summary

This Physics-Informed Neural Networks approach represents a novel application of deep learning to ceramic manufacturing, combining the flexibility of neural networks with the rigor of physics-based modeling. The method addresses key industrial challenges including sparse sensor data, sensor failures, and the need for real-time, physics-consistent temperature predictions.

**Key Innovation:** Integration of heat transfer physics directly into the neural network loss function, enabling accurate interpolation even with limited sensor data while maintaining physical consistency and providing uncertainty quantification for quality control.
