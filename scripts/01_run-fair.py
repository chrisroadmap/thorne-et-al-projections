# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # run fair-2.1.3 with calibration 1.4.4

# %%
output_ensemble_size = 841

# %%
import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import pooch
import xarray as xr

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

print("Running SSP scenarios...")

scenarios = [
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
]

df_solar = pd.read_csv(
    "../input/solar_erf_timebounds.csv", index_col="year"
)
df_volcanic = pd.read_csv(
    "../input/volcanic_ERF_1750-2101_timebounds.csv",
    index_col="timebounds",
)

solar_forcing = np.zeros(352)
volcanic_forcing = np.zeros(352)
volcanic_forcing[:352] = df_volcanic["erf"].loc[1750:2101].values
solar_forcing = df_solar["erf"].loc[1750:2101].values

df_methane = pd.read_csv(
    "../input/CH4_lifetime.csv", index_col=0,
)
df_configs = pd.read_csv(
    "../input/calibrated_constrained_parameters.csv", index_col=0,
)
df_landuse = pd.read_csv(
    "../input/landuse_scale_factor.csv", index_col=0,
)
df_lapsi = pd.read_csv(
    "../input/lapsi_scale_factor.csv", index_col=0,
)
valid_all = df_configs.index

trend_shape = np.ones(352)
trend_shape[:271] = np.linspace(0, 1, 271)

f = FAIR(ch4_method="Thornhill2021")
f.define_time(1750, 2101, 1)
f.define_scenarios(scenarios)
f.define_configs(valid_all)
species, properties = read_properties()
species.remove("Halon-1202")
species.remove("NOx aviation")
species.remove("Contrails")
f.define_species(species, properties)
f.allocate()

# run with harmonized emissions
da_emissions = xr.load_dataarray(
    "../input/ssps_harmonized_1750-2499.nc"
)

da = da_emissions.loc[dict(config="unspecified")][:351, ...]
fe = da.expand_dims(dim=["config"], axis=(2))
f.emissions = fe.drop("config") * np.ones((1, 1, output_ensemble_size, 1))

# solar and volcanic forcing
fill(
    f.forcing,
    volcanic_forcing[:, None, None] * df_configs["fscale_Volcanic"].values.squeeze(),
    specie="Volcanic",
)
fill(
    f.forcing,
    solar_forcing[:, None, None] * df_configs["fscale_solar_amplitude"].values.squeeze()
    + trend_shape[:, None, None] * df_configs["fscale_solar_trend"].values.squeeze(),
    specie="Solar",
)

# climate response
fill(
    f.climate_configs["ocean_heat_capacity"],
    df_configs.loc[:, "clim_c1":"clim_c3"].values,
)
fill(
    f.climate_configs["ocean_heat_transfer"],
    df_configs.loc[:, "clim_kappa1":"clim_kappa3"].values,
)  # not massively robust, since relies on kappa1, kappa2, kappa3 being in adjacent cols
fill(
    f.climate_configs["deep_ocean_efficacy"],
    df_configs["clim_epsilon"].values.squeeze(),
)
fill(
    f.climate_configs["gamma_autocorrelation"],
    df_configs["clim_gamma"].values.squeeze(),
)
fill(f.climate_configs["sigma_eta"], df_configs["clim_sigma_eta"].values.squeeze())
fill(f.climate_configs["sigma_xi"], df_configs["clim_sigma_xi"].values.squeeze())
fill(f.climate_configs["seed"], df_configs["seed"])
fill(f.climate_configs["stochastic_run"], True)
fill(f.climate_configs["use_seed"], True)
fill(f.climate_configs["forcing_4co2"], df_configs["clim_F_4xCO2"])

# species level
f.fill_species_configs()

# carbon cycle
fill(f.species_configs["iirf_0"], df_configs["cc_r0"].values.squeeze(), specie="CO2")
fill(
    f.species_configs["iirf_airborne"],
    df_configs["cc_rA"].values.squeeze(),
    specie="CO2",
)
fill(
    f.species_configs["iirf_uptake"], df_configs["cc_rU"].values.squeeze(), specie="CO2"
)
fill(
    f.species_configs["iirf_temperature"],
    df_configs["cc_rT"].values.squeeze(),
    specie="CO2",
)

# aerosol indirect
fill(f.species_configs["aci_scale"], df_configs["aci_beta"].values.squeeze())
fill(
    f.species_configs["aci_shape"],
    df_configs["aci_shape_so2"].values.squeeze(),
    specie="Sulfur",
)
fill(
    f.species_configs["aci_shape"],
    df_configs["aci_shape_bc"].values.squeeze(),
    specie="BC",
)
fill(
    f.species_configs["aci_shape"],
    df_configs["aci_shape_oc"].values.squeeze(),
    specie="OC",
)

# methane lifetime baseline and sensitivity
fill(
    f.species_configs["unperturbed_lifetime"],
    df_methane.loc["historical_best", "base"],
    specie="CH4",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "CH4"],
    specie="CH4",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "N2O"],
    specie="N2O",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "VOC"],
    specie="VOC",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "NOx"],
    specie="NOx",
)
fill(
    f.species_configs["ch4_lifetime_chemical_sensitivity"],
    df_methane.loc["historical_best", "HC"],
    specie="Equivalent effective stratospheric chlorine",
)
fill(
    f.species_configs["lifetime_temperature_sensitivity"],
    df_methane.loc["historical_best", "temp"],
)

# correct land use  and LAPSI scale factor terms
fill(
    f.species_configs["land_use_cumulative_emissions_to_forcing"],
    df_landuse.loc["historical_best", "CO2_AFOLU"],
    specie="CO2 AFOLU",
)
fill(
    f.species_configs["lapsi_radiative_efficiency"],
    df_lapsi.loc["historical_best", "BC"],
    specie="BC",
)

# emissions adjustments for N2O and CH4 (we don't want to make these defaults as people
# might wanna run pulse expts with these gases)
fill(f.species_configs["baseline_emissions"], 38.246272, specie="CH4")
fill(f.species_configs["baseline_emissions"], 0.92661989, specie="N2O")
fill(f.species_configs["baseline_emissions"], 19.41683292, specie="NOx")
fill(f.species_configs["baseline_emissions"], 2.293964929, specie="Sulfur")
fill(f.species_configs["baseline_emissions"], 348.4549732, specie="CO")
fill(f.species_configs["baseline_emissions"], 60.62284009, specie="VOC")
fill(f.species_configs["baseline_emissions"], 2.096765609, specie="BC")
fill(f.species_configs["baseline_emissions"], 15.44571911, specie="OC")
fill(f.species_configs["baseline_emissions"], 6.656462698, specie="NH3")
fill(f.species_configs["baseline_emissions"], 0.92661989, specie="N2O")
fill(f.species_configs["baseline_emissions"], 0.02129917, specie="CCl4")
fill(f.species_configs["baseline_emissions"], 202.7251231, specie="CHCl3")
fill(f.species_configs["baseline_emissions"], 211.0095537, specie="CH2Cl2")
fill(f.species_configs["baseline_emissions"], 4544.519056, specie="CH3Cl")
fill(f.species_configs["baseline_emissions"], 111.4920237, specie="CH3Br")
fill(f.species_configs["baseline_emissions"], 0.008146006, specie="Halon-1211")
fill(f.species_configs["baseline_emissions"], 0.000010554155, specie="SO2F2")
fill(f.species_configs["baseline_emissions"], 0, specie="CF4")

# aerosol direct
for specie in [
    "BC",
    "CH4",
    "N2O",
    "NH3",
    "NOx",
    "OC",
    "Sulfur",
    "VOC",
    "Equivalent effective stratospheric chlorine",
]:
    fill(
        f.species_configs["erfari_radiative_efficiency"],
        df_configs[f"ari_{specie}"],
        specie=specie,
    )

# forcing scaling
for specie in [
    "CO2",
    "CH4",
    "N2O",
    "Stratospheric water vapour",
    "Light absorbing particles on snow and ice",
    "Land use",
]:
    fill(
        f.species_configs["forcing_scale"],
        df_configs[f"fscale_{specie}"].values.squeeze(),
        specie=specie,
    )

for specie in [
    "CFC-11",
    "CFC-12",
    "CFC-113",
    "CFC-114",
    "CFC-115",
    "HCFC-22",
    "HCFC-141b",
    "HCFC-142b",
    "CCl4",
    "CHCl3",
    "CH2Cl2",
    "CH3Cl",
    "CH3CCl3",
    "CH3Br",
    "Halon-1211",
    "Halon-1301",
    "Halon-2402",
    "CF4",
    "C2F6",
    "C3F8",
    "c-C4F8",
    "C4F10",
    "C5F12",
    "C6F14",
    "C7F16",
    "C8F18",
    "NF3",
    "SF6",
    "SO2F2",
    "HFC-125",
    "HFC-134a",
    "HFC-143a",
    "HFC-152a",
    "HFC-227ea",
    "HFC-23",
    "HFC-236fa",
    "HFC-245fa",
    "HFC-32",
    "HFC-365mfc",
    "HFC-4310mee",
]:
    fill(
        f.species_configs["forcing_scale"],
        df_configs["fscale_minorGHG"].values.squeeze(),
        specie=specie,
    )

# ozone
for specie in [
    "CH4",
    "N2O",
    "Equivalent effective stratospheric chlorine",
    "CO",
    "VOC",
    "NOx",
]:
    fill(
        f.species_configs["ozone_radiative_efficiency"],
        df_configs[f"o3_{specie}"],
        specie=specie,
    )

# tune down volcanic efficacy
fill(f.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")


# initial condition of CO2 concentration (but not baseline for forcing calculations)
fill(
    f.species_configs["baseline_concentration"],
    df_configs["cc_co2_concentration_1750"].values.squeeze(),
    specie="CO2",
)

# initial conditions
initialise(f.concentration, f.species_configs["baseline_concentration"])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

f.run()

# %%
f.forcing

# %%
f.species

# %%
f.species[53]

# %%
f.forcing_sum.sel(scenario='ssp245').median(dim='config')

# %%
import matplotlib.pyplot as pl

# %%
pl.plot(0.5 * 
    (
        (f.forcing_sum[1:, 2, :].data- f.forcing[1:,2,:,52:54].sum(dim='specie').data) +
        (f.forcing_sum[:-1, 2, :].data - f.forcing[:-1,2,:,52:54].sum(dim='specie').data)
    )
);

# %%
anthro = (0.5 * 
    (
        (f.forcing_sum[1:, :, :].data- f.forcing[1:,:,:,52:54].sum(dim='specie').data) +
        (f.forcing_sum[:-1, :, :].data - f.forcing[:-1,:,:,52:54].sum(dim='specie').data)
    )
)

# %%
pl.plot(0.5 * (
        (f.forcing[1:,2,:,52:54].sum(dim='specie').data) +
        (f.forcing[:-1,2,:,52:54].sum(dim='specie').data)
    )
);

# %%
natural = (0.5 * (
        (f.forcing[1:,:,:,52:54].sum(dim='specie').data) +
        (f.forcing[:-1,:,:,52:54].sum(dim='specie').data)
    )
)

# %%
pl.plot(0.5 * 
    (
        (f.forcing_sum[1:, 2, :].data) +
        (f.forcing_sum[:-1, 2, :].data)
    )
);

# %%
total = (0.5 * 
    (
        (f.forcing_sum[1:, :, :].data) +
        (f.forcing_sum[:-1, :, :].data)
    )
)

# %%
pl.plot(0.5 * 
    (
        (f.forcing[1:,2,:,54:56].sum(dim='specie').data) +
        (f.forcing[:-1,2,:,54:56].sum(dim='specie').data)
    )
);

# %%
aerosol = (0.5 * 
    (
        (f.forcing[1:,:,:,54:56].sum(dim='specie').data) +
        (f.forcing[:-1,:,:,54:56].sum(dim='specie').data)
    )
)

# %%
pl.plot(0.5 * 
    (
        (f.forcing[1:,2,:,2:5].sum(dim='specie').data + f.forcing[1:,2,:,11:52].sum(dim='specie').data) +
        (f.forcing[:-1,2,:,2:5].sum(dim='specie').data + f.forcing[:-1,2,:,11:52].sum(dim='specie').data)
    )
);

# %%
ghg = (0.5 * 
    (
        (f.forcing[1:,:,:,2:5].sum(dim='specie').data + f.forcing[1:,:,:,11:52].sum(dim='specie').data) +
        (f.forcing[:-1,:,:,2:5].sum(dim='specie').data + f.forcing[:-1,:,:,11:52].sum(dim='specie').data)
    )
)

# %%
pl.plot(0.5 * 
    (
        (f.forcing[1:,2,:,56:].sum(dim='specie').data) +
        (f.forcing[:-1,2,:,56:].sum(dim='specie').data)
    )
);

# %%
other = (0.5 * 
    (
        (f.forcing[1:,:,:,56:].sum(dim='specie').data) +
        (f.forcing[:-1,:,:,56:].sum(dim='specie').data)
    )
);

# %%
np.max(anthro - (other + ghg + aerosol))

# %%
np.max(np.abs(total - (anthro + natural)))

# %%
total.shape

# %%
ds = xr.Dataset(
    data_vars = dict(
        total = (['time', 'scenario', 'config'], total),
        anthro = (['time', 'scenario', 'config'], anthro),
        natural = (['time', 'scenario', 'config'], natural),
        ghg = (['time', 'scenario', 'config'], ghg),
        aerosol = (['time', 'scenario', 'config'], aerosol),
        other = (['time', 'scenario', 'config'], other),
    ),
    coords = dict(
        scenario = scenarios,
        time = np.arange(1750.5, 2101),
        config = df_configs.index
    ),
    attrs = dict(units = 'W / m2')
)

# %%
os.makedirs('../output', exist_ok=True)

# %%
ds.to_netcdf('../output/ssp_forcing_fair2.1.3_cal1.4.5.nc')

# %%
