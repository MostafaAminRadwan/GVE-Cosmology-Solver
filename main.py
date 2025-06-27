import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology

# --- Part 1: Data and Model Definitions ---

# Cosmic Chronometer Data (32 points from Moresco et al. 2022, widely used)
z_cc = np.array([0.07, 0.09, 0.12, 0.17, 0.1791, 0.1993, 0.2, 0.27, 0.28, 0.3519, 0.3802, 0.4, 0.4004, 0.4247, 0.4497, 0.47, 0.4783, 0.48, 0.5929, 0.6797, 0.7812, 0.8754, 0.88, 0.9, 1.037, 1.3, 1.363, 1.43, 1.53, 1.75, 1.965, 2.34])
h_cc = np.array([69.0, 69.0, 68.6, 83.0, 75.0, 75.0, 72.9, 77.0, 88.8, 83.0, 83.0, 95.0, 77.0, 87.1, 92.8, 89.0, 80.0, 97.0, 104.0, 92.0, 105.0, 125.0, 17.0, 40.0, 23.0, 13.0, 17.0, 33.6, 18.0, 14.0, 26.0, 50.4])
err_cc = np.array([19.6, 12.0, 26.2, 8.0, 4.0, 5.0, 29.6, 14.0, 36.6, 14.0, 13.5, 17.0, 10.2, 11.2, 12.9, 34.0, 9.0, 62.0, 13.0, 8.0, 12.0, 17.0, 40.0, 23.0, 13.0, 17.0, 33.6, 18.0, 14.0, 26.0, 50.4, 48.])

# Planck 2018 Reference Model
planck18_params = {
    'flat': True, 'H0': 67.36, 'Om0': 0.3153, 'Ob0': 0.0493,
    'sigma8': 0.8111, 'ns': 0.9649,
}

# --- CORRECTED SECTION ---
# Optimal GVE Attractor Model, defined using Ode0 instead of Ok0
gve_params = {
    'H0': 69.42,
    'Om0': 0.24955,
    'Ode0': 0.65301, # Specify Dark Energy density, Ok0 is derived from this
    'Ob0': planck18_params['Ob0'],
    'ns': planck18_params['ns'],
    'sigma8': planck18_params['sigma8'],
    'flat': False, # Explicitly state the cosmology is not flat
}
# --- END CORRECTED SECTION ---


# --- Part 2: Generate and Plot Figure 1 (H(z) Fit) ---

print("Generating H(z) fit plot...")

plt.style.use('seaborn-v0_8-ticks')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

z_plot = np.linspace(0, 2.5, 400)

# Calculate H(z) for Planck 2018
planck_object = cosmology.setCosmology('planck18_ref', planck18_params)
hz_planck = planck_object.Hz(z_plot)

# Calculate H(z) for GVE model
# Note: Since the GVE model's energy density is not a simple lambda,
# using 'Ode0' here for background evolution is an approximation.
# However, for plotting H(z), it's a very good one as shown in your paper.
gve_object_for_hz = cosmology.setCosmology('gve_model_for_plotting', gve_params)
hz_gve = gve_object_for_hz.Hz(z_plot)

# Main panel
ax1.plot(z_plot, hz_gve, color='red', linestyle='-', label=r'GVE Attractor ($H_0 \approx 69.4$)')
ax1.plot(z_plot, hz_planck, color='blue', linestyle='--', label=r'Planck 2018 $\Lambda$CDM ($H_0 \approx 67.4$)')
ax1.errorbar(z_cc, h_cc, yerr=err_cc, fmt='o', color='black', ecolor='gray', capsize=3, label='Cosmic Chronometers')
ax1.set_ylabel(r'$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(linestyle=':')
ax1.tick_params(axis='both', which='major', labelsize=12)

# Residuals panel
h_gve_interp = np.interp(z_cc, z_plot, hz_gve)
residuals = h_cc - h_gve_interp
ax2.errorbar(z_cc, residuals, yerr=err_cc, fmt='o', color='red', ecolor='lightcoral', capsize=3)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_xlabel(r'Redshift ($z$)', fontsize=14)
ax2.set_ylabel(r'$\Delta H(z)$', fontsize=14)
ax2.grid(linestyle=':')
ax2.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xlim(0, max(z_plot))

plt.tight_layout()
plt.savefig('Hz_fit_plot.png', dpi=300, bbox_inches='tight')
print("Saved Hz_fit_plot.png")


# --- Part 3: Generate and Plot Figure 2 (Growth Comparison) ---

print("Generating growth comparison plot...")

# For a fair comparison of the growth *shape*, create a flat LCDM with GVE's Om0
lcdm_ref_params = {
    'H0': gve_params['H0'], 'Om0': gve_params['Om0'], 'Ob0': gve_params['Ob0'],
    'ns': gve_params['ns'], 'sigma8': gve_params['sigma8'], 'flat': True,
}

plt.figure(figsize=(8, 6))
z_growth_plot = np.linspace(0, 5, 400)
z_high = 499.0

# Calculate and normalize growth for the reference LCDM
cosmo_ref = cosmology.setCosmology('lcdm_ref_for_growth', lcdm_ref_params)
D_ref = cosmo_ref.growthFactor(z_growth_plot)
D_ref_norm = cosmo_ref.growthFactor(z_high)
plt.plot(z_growth_plot, D_ref / D_ref_norm, color='darkorange', linestyle='--', label=r'Reference Flat $\Lambda$CDM')

# Calculate and normalize growth for the GVE model
cosmo_gve = cosmology.setCosmology('gve_model_for_growth', gve_params)
D_gve = cosmo_gve.growthFactor(z_growth_plot)
D_gve_norm = cosmo_gve.growthFactor(z_high)
plt.plot(z_growth_plot, D_gve / D_gve_norm, color='blue', linestyle='-', label='GVE Attractor Model')

plt.xlabel(r'Redshift ($z$)', fontsize=14)
plt.ylabel(r'Normalized Growth Factor, $D(z)/D(z_{\text{high}})$', fontsize=14)
plt.title('Comparison of Linear Structure Growth History', fontsize=16)
plt.legend(fontsize=12)
plt.grid(linestyle=':')
plt.xlim(0, 5)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('growth_comparison_plot.png', dpi=300, bbox_inches='tight')
print("Saved growth_comparison_plot.png")

print("\nDone. Both plot files have been generated.")
