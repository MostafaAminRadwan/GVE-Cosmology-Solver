import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the GVE Model Parameters for the Best-Fit Solution
class GVEModel:
    def __init__(self):
        # Constants in SI units
        self.G_N = 6.67430e-11
        self.MPC_TO_M = 3.08567758e22
        self.H0_s_inv = (69.77 * 1000) / self.MPC_TO_M
        self.rho_crit_0 = 3 * self.H0_s_inv**2 / (8 * np.pi * self.G_N)
        
        # Cosmological density parameters
        self.Omega_m0 = 0.24955
        self.Omega_k0 = 0.09735
        self.Omega_r0 = 9.0e-5
        
        # Potential parameters for the best-fit model
        self.phi_c = 6.0
        self.v0 = 7.0e-27
        self.e1, self.a1 = 0.60, 1.5
        self.e2, self.a2 = 0.47, 0.7
        self.e3, self.l = 0.60, 0.28
        self.e4, self.a4 = 0.1, 15.0

    def potential(self, phi):
        original = (1 + self.e1*np.exp(-self.a1*phi) + 
             self.e2*np.exp(-self.a2*phi) + 
             self.e3*np.exp(-self.l*(phi-self.phi_c)**2))
        early = self.e4 * np.exp(-self.a4 * phi)
        return self.v0 * (original + early)

    def potential_prime(self, phi):
        original_prime = (-self.a1*self.e1*np.exp(-self.a1*phi) - 
             self.a2*self.e2*np.exp(-self.a2*phi) - 
 2*self.l*self.e3*(phi-self.phi_c)*np.exp(-self.l*(phi-self.phi_c)**2))   
early_prime = -self.a4*self.e4*np.exp(-self.a4*phi)
        return self.v0 * (original_prime + early_prime)

    def odesystem(self, N, Y):
        phi, psi = Y
        a = np.exp(N)
        
        rho_m = self.Omega_m0 * self.rho_crit_0 * a**-3
        rho_r = self.Omega_r0 * self.rho_crit_0 * a**-4
        V_phi = self.potential(phi)
        
        k_term = self.Omega_k0 * self.H0_s_inv**2 * a**-2
        A = (8 * np.pi * self.G_N / 3.0)
        B = A / 2.0
        
        num_H2 = A * (rho_m + rho_r + V_phi) - k_term
        den_H2 = 1 - B * psi**2
        if den_H2 <= 1e-12 or num_H2 <= 0: return [0, 0]
        H_sq = num_H2 / den_H2
        
        V_prime = self.potential_prime(phi)
        dphi_dN = psi
        dpsi_dN = -3 * psi - V_prime / H_sq
        return [dphi_dN, dpsi_dN]

# --- Main execution ---
model = GVEModel()
N_start, N_end = -15, 0
y_initial = [0.0, 0.0]
N_eval = np.linspace(N_start, N_end, 500)

# Solve the ODE system
solution = solve_ivp(model.odesystem, (N_start, N_end), y_initial, 
                     method='Radau', t_eval=N_eval)

# Post-process to get H(z)
N_sol, Y_sol = solution.t, solution.y
phi_sol, psi_sol = Y_sol
a_sol = np.exp(N_sol)
z_sol = 1.0/a_sol - 1.0

# Correctly calculate H for each point in the solution
H_values_si = []
for i in range(len(N_sol)):
    # Call odesystem again just to get H_sq, which is a bit inefficient but clear
    # A more optimized code would have odesystem return H_sq as well
    phi, psi = Y_sol[:, i]
    a = np.exp(N_sol[i])
    rho_m = model.Omega_m0 * model.rho_crit_0 * a**-3
    rho_r = model.Omega_r0 * model.rho_crit_0 * a**-4
    V_phi = model.potential(phi)
    k_term = model.Omega_k0 * model.H0_s_inv**2 * a**-2
    A = (8*np.pi*model.G_N/3.0); B = A/2.0
    num = A*(rho_m + rho_r + V_phi) - k_term
    den = 1 - B*psi**2
    H_sq = num/den if den > 1e-12 else 0
    H_values_si.append(np.sqrt(H_sq) if H_sq > 0 else 0)

H_kms_mpc = np.array(H_values_si) * model.MPC_TO_M / 1000

# Plot the result
plt.figure(figsize=(10,6))
plt.plot(z_sol, H_kms_mpc, label='GVE Best-Fit Model H(z)')
plt.xlabel('Redshift z')
plt.ylabel('Hubble Parameter H(z) [km/s/Mpc]')
plt.title('Minimal GVE Simulation: Expansion History')
plt.legend()
plt.grid(True)
plt.show()
