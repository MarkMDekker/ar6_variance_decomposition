# ======================================== #
# Class that does the decomposition
# Input: files produced by class_datahandling.py
# Output: XRvar.nc, which contains the variance fractions explained by the three drivers
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import xarray as xr
import yaml

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class decomposition(object):

    # =========================================================== #
    # =========================================================== #

    def __init__(self):
        print("# ==================================== #")
        print("# Initializing decomposition class     #")
        print("# ==================================== #")

        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            diction = yaml.load(file, Loader=yaml.FullLoader)
        self.location_ipccdata = Path(diction['location_ipcc'])
        self.save = diction['save']
        self.sample_size_per_ms = int(diction['sample_size_per_ms'])
        self.resampling = int(diction['resampling'])
        self.removal_c8 = diction['removal_c8']
        self.generate_composites = diction['generate_composites']

        # Read in data produced by the code in data_handling.py
        self.XRmeta = xr.open_dataset(self.current_dir / "Data" / "Handling_files" / ("XRmeta.nc"))
        self.XRar6 = xr.open_dataset(self.current_dir / "Data" / "Handling_files" / ("XRdata.nc"))
        self.years = list(self.XRar6.Time.data)
        self.var_all = np.array(self.XRar6.Variable)
        with open(self.current_dir / "Data" / "Handling_files" / 'XRsubs.pickle', 'rb') as handle:
            self.XRsubs = pickle.load(handle)

    # =========================================================== #
    # =========================================================== #
    
    def sampling_and_decomposing(self, printy='on'):
        if printy=='on': print("- Generate samples and apply decomposition")
        
        def generate_listoflists(self, xrsub):
            xrmeta = self.XRmeta
            modscens = np.array(xrsub.ModelScenario)
            mods = np.array([i.split('|')[0] for i in modscens])
            ccat = np.array(xrmeta.sel(ModelScenario=xrsub.ModelScenario).Category.data)
            unimods = np.unique(mods)
            uniccat = np.unique(ccat)
            whs = []
            for m_i, m in enumerate(unimods):
                for c_i, c in enumerate(uniccat):
                    wh = np.where((mods == m) & (ccat == c))[0]
                    whs.append(wh)
            return whs

        def generate_samples(self, var):
            xrsub = self.XRsubs[var]
            values = np.array(xrsub.Value)
            values_nn = np.array(xrsub.Value)
            values = values - np.mean(values)
            values = values / np.std(values)
            modscens = np.array(xrsub.ModelScenario)
            mods = np.array([i.split('|')[0] for i in modscens])
            ccat = np.array(self.XRmeta.sel(ModelScenario=xrsub.ModelScenario).Category.data)
            unimods = np.unique(mods)
            uniccat = np.unique(ccat)

            whs = generate_listoflists(self, xrsub)
            ss = self.sample_size_per_ms

            indices = np.zeros(shape=(7, self.resampling, len(self.years)))
            for n_i in range(self.resampling):
                sample1 = np.zeros(shape=(2, len(whs)*ss)).astype(str)
                a = 0
                for m_i, m in enumerate(unimods):
                    for c_i, c in enumerate(uniccat):
                        wh = whs[a]
                        sample1[0][a*ss:a*ss+ss] = [m]*ss
                        sample1[1][a*ss:a*ss+ss] = [c]*ss
                        a+=1
                np.random.shuffle(sample1.T)

                sample2 = np.zeros(shape=(2, len(whs)*ss)).astype(str)
                a = 0
                for m_i, m in enumerate(unimods):
                    for c_i, c in enumerate(uniccat):
                        wh = whs[a]
                        sample2[0][a*ss:a*ss+ss] = [m]*ss
                        sample2[1][a*ss:a*ss+ss] = [c]*ss
                        a+=1
                np.random.shuffle(sample2.T)

                M1 = np.zeros(shape=(len(sample1[0]), len(self.years)))
                M1nn = np.zeros(shape=(len(sample1[0]), len(self.years)))
                M2 = np.zeros(shape=(len(sample1[0]), len(self.years)))
                Nm = np.zeros(shape=(len(sample1[0]), len(self.years)))
                Nc = np.zeros(shape=(len(sample1[0]), len(self.years)))
                Nmc = np.zeros(shape=(len(sample1[0]), len(self.years)))
                cv = np.zeros(shape=(len(sample1[0]), len(self.years)))
                for m in unimods:
                    for c in uniccat:
                        wh = np.where((mods == m) & (ccat == c))[0]
                        if len(wh) > 0:
                            wh1 = np.where((sample1[0] == m) & (sample1[1] == c))[0]
                            wh2 = np.where((sample2[0] == m) & (sample2[1] == c))[0]
                            choice = np.random.choice(wh, self.sample_size_per_ms, replace=True)
                            M1nn[wh1] = values_nn[choice]
                            M1[wh1] = values[choice]
                            M2[wh2] = values[np.random.choice(wh, self.sample_size_per_ms, replace=True)]

                            wh_m = np.where((sample1[0] == m) & (sample2[1] == c))[0]
                            wh_c = np.where((sample2[0] == m) & (sample1[1] == c))[0]
                            Nm[wh_m] = values[np.random.choice(wh, len(wh_m), replace=True)]
                            Nc[wh_c] = values[np.random.choice(wh, len(wh_c), replace=True)]

                            Nmc[wh1] = values[np.random.choice(wh, len(wh1), replace=True)]

                vtot = np.var(M1nn, axis=0) / np.mean(M1nn)
                vtot_norm = np.var(M1, axis=0)
                cv = np.std(M1nn, axis=0) / np.mean(M1nn)
                s_m = np.diag(1/(len(sample1[0])-1)*np.dot(M1.T, Nm) - 1/(len(sample1[0]))*np.dot(M1.T, M2))/np.var(M1, axis=0)
                s_c = np.diag(1/(len(sample1[0])-1)*np.dot(M1.T, Nc) - 1/(len(sample1[0]))*np.dot(M1.T, M2))/np.var(M1, axis=0)
                comb = np.diag(1/(len(sample1[0])-1)*np.dot(M1.T, Nmc) - 1/len(sample1[0])*np.dot(M1.T, M2))/np.var(M1, axis=0)
                s_mc = comb - s_m - s_c
                s_z = 1 - comb
                indices[:, n_i, :] = [vtot, s_m, s_c, s_mc, s_z, vtot_norm, cv]
            self.comb = comb
            return np.mean(indices, axis=1)

        self.variances = np.zeros(shape=(len(self.var_all), 7, len(self.years)))
        if printy == 'on':
            for v_i in tqdm(range(len(self.var_all))):
                var = self.var_all[v_i]
                vtot, s_m, s_c, s_mc, s_z, vtot_norm, cv = generate_samples(self, var)
                self.variances[v_i][0] = vtot
                self.variances[v_i][1] = s_m
                self.variances[v_i][2] = s_c
                self.variances[v_i][3] = s_z
                self.variances[v_i][4] = s_mc
                self.variances[v_i][5] = vtot_norm
                self.variances[v_i][6] = cv
        else:
            for v_i in range(len(self.var_all)):
                var = self.var_all[v_i]
                vtot, s_m, s_c, s_mc, s_z, vtot_norm, cv = generate_samples(self, var)
                self.variances[v_i][0] = vtot
                self.variances[v_i][1] = s_m
                self.variances[v_i][2] = s_c
                self.variances[v_i][3] = s_z
                self.variances[v_i][4] = s_mc
                self.variances[v_i][5] = vtot_norm
                self.variances[v_i][6] = cv
        
        ds = xr.Dataset({"Var_total": (("Variable", "Time"), self.variances[:, 0]),
                        "S_m": (("Variable",  "Time"), self.variances[:, 1]),
                        "S_c": (("Variable", "Time"), self.variances[:, 2]),
                        "S_z": (("Variable", "Time"), self.variances[:, 3]),
                        "S_mc": (("Variable", "Time"), self.variances[:, 4]),
                        "Var_total_norm": (("Variable", "Time"), self.variances[:, 5]),
                        "CoefVar": (("Variable", "Time"), self.variances[:, 6])},
                        coords={
                        "Variable": self.var_all,
                        "Time": self.years})
        self.XRvar = ds

    # =========================================================== #
    # =========================================================== #
    
    def savings(self):
        print("- Save stuff")
        if self.save == 'yes':
            self.XRvar.to_netcdf(self.current_dir / "Data" / "Output_files" / "Variances.nc")
