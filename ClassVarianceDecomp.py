# ======================================== #
# Class to do the AR6 variance decomposition
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import xarray as xr
import yaml

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class variancedecomp(object):

    # =========================================================== #
    # INITIALIZATION
    # =========================================================== #

    def __init__(self):
        print("# ==================================== #")
        print("# Initializing variancedecomp class    #")
        print("# ==================================== #")

        self.current_dir = Path.cwd()
        self.location_ipccdata = Path("X:/user/dekkerm/Data/IPCC")

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            diction = yaml.load(file, Loader=yaml.FullLoader)
        self.varfile = diction['varfile']
        self.save = diction['save']
        self.threshold_dataremoval = int(diction['threshold_dataremoval'])
        self.sample_size_per_ms = int(diction['sample_size_per_ms'])
        # self.times_sampling = int(diction['times_sampling'])
        # self.combined_sample_size = int(diction['combined_sample_size'])
        self.variable_counting = int(diction['variable_counting'])
        self.add_wind_solar = int(diction['add_wind_solar'])
        self.resampling = int(diction['resampling'])

        # Run subclasses
        print('- Reading variable lists')
        self.vars = variancedecomp.class_vars(self)
        print('- Reading metadata')
        self.meta = variancedecomp.class_meta(self)
        print('- Reading AR6 data')
        self.ar6 = variancedecomp.class_ar6(self)
        print('- Applying sampled decomposition')
        self.decomp = variancedecomp.class_decomposition(self)
        print('- Done!')

    # =========================================================== #
    # Subclass variable lists (input)
    # =========================================================== #

    class class_vars(object):
        def __init__(self, mainclass):
            self.path = mainclass.current_dir / "Data" / mainclass.varfile
            xls = pd.read_excel(self.path, sheet_name = "Data")
            vars = np.array(xls["Variable"])
            cats = np.array(xls["Category"])
            #Names = list(xls.keys())
            #Vlists = []
            #FullVarlist = []
            #for i in range(len(Names)):
            #    Vlists.append(list(pd.read_excel(self.path, sheet_name = Names[i]).Variable))
            #    FullVarlist = FullVarlist+list(pd.read_excel(self.path, sheet_name = Names[i]).Variable)
            #self.full = FullVarlist+['Primary Energy']
            #self.listed = Vlists
            #self.listnames = Names
            self.listnames = np.unique(cats)
            self.full = vars
            self.listed = []
            for c in self.listnames:
                self.listed.append(list(vars[cats == c]))

    # =========================================================== #
    # Subclass meta data
    # =========================================================== #
    
    class class_meta(object):
        def __init__(self, mainclass):
            #self.path = mainclass.location_ipccdata / "ar6_full_metadata_indicators2021_10_14_v3.xlsx"
            self.path = mainclass.location_ipccdata / "AR6_Scenarios_Database_metadata_indicators_v1.0.xlsx"
            DF = pd.read_excel(self.path, sheet_name='meta_Ch3vetted_withclimate')
            idx = np.where((DF.Scenario == 'EN_NPi2020_800') & (DF.Model == 'WITCH 5.0'))[0]
            DF = DF.drop(idx)
            DF = DF.reset_index(drop=True)
            idx = np.where((DF.Scenario == 'EN_NPi2020_900') & (DF.Model == 'WITCH 5.0'))[0]
            DF = DF.drop(idx)
            DF = DF.reset_index(drop=True)
            DF = DF[DF.Vetting_historical == 'Pass']
            self.DF = DF.reset_index(drop=True)
            self.model = np.array(DF.Model)
            self.scen = np.array(DF.Scenario)
            self.modscen = np.array([self.model[i]+'|'+self.scen[i] for i in range(len(DF))])

    # =========================================================== #
    # Subclass actual AR6 data
    # =========================================================== #
    
    class class_ar6(object):
        def __init__(self, mainclass):
            #self.path = mainclass.location_ipccdata / "snapshot_world_with_key_climate_iamc_ar6_2021_10_14.csv"
            self.path = mainclass.location_ipccdata / "AR6_Scenarios_Database_World_v1.0.csv"

            if mainclass.variable_counting == 1:
                # ================================================================================ #
                print("     Saving counts of variables in total database (for reference later)")
                # ================================================================================ #
                DF = pd.read_csv(self.path)
                Mods_raw = np.array(DF.Model)
                Scen_raw = np.array(DF.Scenario)
                DF['ModelScenario'] = [Mods_raw[i]+'|'+Scen_raw[i] for i in range(len(Scen_raw))]
                DF = DF[DF.ModelScenario.isin(mainclass.meta.modscen)]
                DF = DF.reset_index()[DF.keys()]
                Mods_raw = np.array(DF.Model)
                Scen_raw = np.array(DF.Scenario)
                Vars_raw = np.array(DF.Variable)
                modscen = np.array([Mods_raw[i]+'|'+Scen_raw[i] for i in range(len(Scen_raw))])
                d = {}
                d['Variable'] = list(np.unique(Vars_raw))+['Primary Energy|Wind+Solar']
                count = []
                for i in np.unique(Vars_raw):
                    count.append(len(np.unique(modscen[Vars_raw == i])))
                count.append(len(np.intersect1d(np.unique(modscen[Vars_raw == 'Primary Energy|Wind']),
                                                np.unique(modscen[Vars_raw == 'Primary Energy|Solar']))))
                d['Count'] = count
                self.countsdf = pd.DataFrame(d)

            # ================================================================================ #
            print("     Now actual reading of data, and applying various filters to the database")
            # ================================================================================ #
            DF = pd.read_csv(self.path)
            Mods_raw = np.array(DF.Model)
            Scen_raw = np.array(DF.Scenario)
            DF['ModelScenario'] = [Mods_raw[i]+'|'+Scen_raw[i] for i in range(len(Scen_raw))]
            DF = DF[DF.ModelScenario.isin(mainclass.meta.modscen)]
            DF = DF.reset_index()[DF.keys()]
            DF = DF[DF.Variable.isin(mainclass.vars.full)]
            DF = DF.reset_index(drop=True)
            DF = DF.drop(columns=['Region', 'Unit'])
            DF = DF.drop(columns=['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002',
                                  '2003', '2004', '2005', '2006', '2007', '2008', '2009'])
            Mods_raw = np.array(DF.Model)
            Scens = np.array(DF.Scenario)
            ModScen_raw = [Mods_raw[i]+'|'+Scens[i] for i in range(len(DF))]
            wh = []
            for i in range(len(ModScen_raw)):
                if ModScen_raw[i] in mainclass.meta.modscen:
                    wh.append(i)
            DF = DF[DF.index.isin(wh)]
            DF = DF.reset_index(drop=True)
            self.DF = DF
            self.model = np.array(DF.Model)
            self.scen = np.array(DF.Scenario)
            self.vars = np.array(DF.Variable)

            # ================================================================================ #
            print("     Merge model versions")
            # ================================================================================ #
            pathmodelversions = mainclass.location_ipccdata / "ModelConversion.xlsx"
            self.modelversionsdf = pd.read_excel(pathmodelversions, sheet_name='Sheet1')
            modelversions_model = np.array(self.modelversionsdf['Model'])
            for v_i, vers in enumerate(self.modelversionsdf['Model version']):
                mod = modelversions_model[v_i]
                mainclass.meta.model[mainclass.meta.model == vers] = mod
                self.model[self.model == vers] = mod
            mainclass.meta.DF['model'] = mainclass.meta.model
            self.DF['Model'] = self.model
            mainclass.meta.DF['ModelScenario'] = [mainclass.meta.model[i]+'|'+mainclass.meta.scen[i] for i in range(len(mainclass.meta.scen))]
            mainclass.meta.DF = mainclass.meta.DF.drop(columns=['Model', 'Scenario'])
            self.modscen = np.array([self.model[i]+'|'+self.scen[i] for i in range(len(self.scen))])
            self.DF['ModelScenario'] = self.modscen
            self.DF = self.DF[['ModelScenario', 'Variable', '2010', '2011', '2012', '2013', '2014',
                                '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023',
                                '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032',
                                '2033', '2034', '2035', '2036', '2037', '2038', '2039', '2040', '2041',
                                '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049', '2050',
                                '2051', '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059',
                                '2060', '2061', '2062', '2063', '2064', '2065', '2066', '2067', '2068',
                                '2069', '2070', '2071', '2072', '2073', '2074', '2075', '2076', '2077',
                                '2078', '2079', '2080', '2081', '2082', '2083', '2084', '2085', '2086',
                                '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095',
                                '2096', '2097', '2098', '2099', '2100']]

            if mainclass.add_wind_solar == 1:
                # ================================================================================ #
                print("     Add Primary Energy|Wind+Solar") # This should be optimized at some point
                # ================================================================================ #
                ms1 = np.unique(self.modscen[self.vars == 'Primary Energy|Wind'])
                ms2 = np.unique(self.modscen[self.vars == 'Primary Energy|Solar'])
                ms = np.unique(np.intersect1d(ms1, ms2))
                a = 0
                for s in ms:
                    d = {}
                    d['ModelScenario'] = [s]
                    d['Variable'] = ['Primary Energy|Wind+Solar']
                    df1 = self.DF[(self.modscen == s) & (self.vars == 'Primary Energy|Wind')]
                    df2 = self.DF[(self.modscen == s) & (self.vars == 'Primary Energy|Solar')]
                    for t in np.arange(2010, 2101):
                        d[str(t)] = [list(df1[str(t)])[0] + list(df2[str(t)])[0]]
                    if a == 0:
                        DS = pd.DataFrame(d)
                    else:
                        DS = DS.append(pd.DataFrame(d), ignore_index=True)
                    a += 1
                self.DF = self.DF.append(DS)
                self.DF = self.DF.reset_index()
                self.DF = self.DF[self.DF.keys()[1:]]
                self.vars = np.array(self.DF.Variable)
                self.modscen = np.array(self.DF.ModelScenario)

                # ================================================================================ #
                print("     Add Secondary Energy|Electricity|Wind+Solar") # This should be optimized at some point
                # ================================================================================ #

                ms1 = np.unique(self.modscen[self.vars == 'Secondary Energy|Electricity|Wind'])
                ms2 = np.unique(self.modscen[self.vars == 'Secondary Energy|Electricity|Solar'])
                ms = np.unique(np.intersect1d(ms1, ms2))
                a = 0
                for s in ms:
                    d = {}
                    d['ModelScenario'] = [s]
                    d['Variable'] = ['Secondary Energy|Electricity|Wind+Solar']
                    df1 = self.DF[(self.modscen == s) & (self.vars == 'Secondary Energy|Electricity|Wind')]
                    df2 = self.DF[(self.modscen == s) & (self.vars == 'Secondary Energy|Electricity|Solar')]
                    for t in np.arange(2010, 2101):
                        d[str(t)] = [list(df1[str(t)])[0] + list(df2[str(t)])[0]]
                    if a == 0:
                        DS = pd.DataFrame(d)
                    else:
                        DS = DS.append(pd.DataFrame(d), ignore_index=True)
                    a += 1
                self.DF = self.DF.append(DS)
                self.DF = self.DF.reset_index()
                self.DF = self.DF[self.DF.keys()[1:]]
            self.model = np.array(DF.Model)
            self.scen = np.array(DF.Scenario)
            self.vars = np.array(DF.Variable)
            self.modscen = np.array([self.model[i]+'|'+self.scen[i] for i in range(len(self.scen))])

            # ================================================================================ #
            print("     Convert time dimension and Xarray")
            # ================================================================================ #

            XRdummy = mainclass.meta.DF.set_index(['ModelScenario'])
            mainclass.meta.XR = xr.Dataset.from_dataframe(XRdummy)

            DF_timadj = self.DF.melt(id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value")
            DF_timadj['Time'] = np.array(DF_timadj['Time']).astype(int)
            dfdummy = DF_timadj.set_index(['ModelScenario', 'Variable', 'Time'])
            self.XR = xr.Dataset.from_dataframe(dfdummy)

            # ================================================================================ #
            print("     Temporal interpolation")
            # ================================================================================ #

            self.XR = self.XR.sel(Variable=mainclass.vars.full)
            self.XR = self.XR.interpolate_na(dim="Time", method="linear")

            # ================================================================================ #
            print("     Save stuff")
            # ================================================================================ #

            if mainclass.save == 'yes':
                try:
                    self.countsdf.to_csv(mainclass.current_dir / "Data" / "Counts.csv")
                except:
                    3
                mainclass.meta.XR.to_netcdf(mainclass.current_dir / "Data" / "XRmeta.nc")
                self.XR.to_netcdf(mainclass.current_dir / "Data" / "XRdata.nc")
    
    # =========================================================== #
    # Sublists for information (optional)
    # =========================================================== #\

    def generate_modscencounts(self):

        # Model counts for each mod-scen pair
        Models = [self.ar6.XR.ModelScenario.data[i].split('|')[0] for i in range(len(self.ar6.XR.ModelScenario))]
        DFmods = {}
        unimod = np.unique(Models)
        DFmods['Model'] = ['Total']+list(unimod)
        for v in tqdm(range(len(self.vars.listed)+1)):
            numscens = []
            for m in unimod:
                XR_sub = self.ar6.XR.sel(Variable=([self.vars.full]+self.vars.listed)[v])
                XR_sub = XR_sub.dropna('ModelScenario', how='any')
                mods = np.array([XR_sub.ModelScenario.data[i].split('|')[0] for i in range(len(XR_sub.ModelScenario))])
                numscens.append(len(np.where(mods == m)[0]))
            if v == 0:
                DFmods['Full set'] = [sum(numscens)]+numscens
            else:
                DFmods[self.vars.listnames[v-1]] = [sum(numscens)]+numscens
        DFmods = pd.DataFrame(DFmods)
        DFmods = DFmods.sort_values(by=['Full set'], ascending=False)
        DFmods = DFmods.reset_index(drop=True)
        DFmods.to_csv(self.current_dir / "Data" / "Models.csv")

        # C-category counts for each mod-scen pair
        Ccategories = self.ar6.XR.sel(ModelScenario=self.ar6.XR.ModelScenario).Category.data
        DFc = {}
        unicat = np.unique(Ccategories)
        DFc['Ccategory'] = ['Total']+list(unicat)
        for v in tqdm(range(len(self.vars.listed)+1)):
            numscens = []
            for m in unicat:
                XR_sub = self.ar6.XR.sel(Variable=([self.vars.full]+self.vars.listed)[v])
                XR_sub = XR_sub.dropna('ModelScenario', how='any')
                ccat = np.array(self.meta.XR.sel(ModelScenario=XR_sub.ModelScenario).Category.data)
                numscens.append(len(np.where(ccat == m)[0]))
            if v == 0:
                DFc['Full set'] = [sum(numscens)]+numscens
            else:
                DFc[self.vars.listnames[v-1]] = [sum(numscens)]+numscens
        DFc = pd.DataFrame(DFc)
        DFc = DFc.reset_index(drop=True)
        DFc.to_csv(self.current_dir / "Data" / "Ccategories.csv")

    # =========================================================== #
    # Subclass decomposition calculations
    # =========================================================== #

    class class_decomposition(object):

        def __init__(self, mainclass):
            years = list(mainclass.ar6.XR.Time.data)
            mainclass.years = years
            self.XRsubs = {}
            self.variances = np.zeros(shape=(len(mainclass.vars.full), 6, len(years)))
            for v_i in tqdm(range(len(mainclass.vars.full))):
                var = mainclass.vars.full[v_i]
                xrsub = self.preprocess_xr(var, mainclass)
                vtot, s_m, s_c, s_mc, s_z, vtot_norm = self.generate_samples(var, mainclass, xrsub)
                # for n_i in range(mainclass.times_sampling):
                #     xrsample = xrsub.sel(ModelScenario = samples[n_i])
                #     s_m, s_c, s_z, s_mc = self.decompose_variable(var, mainclass, xrsample)
                self.variances[v_i][0] = vtot
                self.variances[v_i][1] = s_m
                self.variances[v_i][2] = s_c
                self.variances[v_i][3] = s_z
                self.variances[v_i][4] = s_mc
                self.variances[v_i][5] = vtot_norm
            
            ds = xr.Dataset({"Var_total": (("Variable", "Time"), self.variances[:, 0]),
                            "S_m": (("Variable",  "Time"), self.variances[:, 1]),
                            "S_c": (("Variable", "Time"), self.variances[:, 2]),
                            "S_z": (("Variable", "Time"), self.variances[:, 3]),
                            "S_mc": (("Variable", "Time"), self.variances[:, 4]),
                            "Var_total_norm": (("Variable", "Time"), self.variances[:, 5])},
                            coords={
                            "Variable": mainclass.vars.full,
                            "Time": years})
            self.xr = ds
            if mainclass.save == 'yes':
                ds.to_netcdf(mainclass.current_dir / "Data" / "Variances.nc")
        
        def preprocess_xr(self, var, mainclass):
            ''' Checks in place '''
            xrsub = mainclass.ar6.XR.sel(Variable = var)
            xrsub = xrsub.dropna('ModelScenario', how='any')
            mods = np.array([xrsub.ModelScenario.data[i].split('|')[0] for i in range(len(xrsub.ModelScenario))])
            ccat = np.array(mainclass.meta.XR.sel(ModelScenario=xrsub.ModelScenario).Category.data)
            modscen = np.array(xrsub.ModelScenario)
            modscen_new = []
            for i in range(len(mods)):
                m = mods[i]
                c = ccat[i]
                if len(np.where(mods == m)[0]) >= mainclass.threshold_dataremoval and c != "C8":
                    modscen_new.append(modscen[i])
            xrsub = xrsub.sel(ModelScenario = modscen_new)
            self.XRsubs[var] = xrsub
            # var_totalraw = np.zeros(len(mainclass.years))+np.nan
            # for y_i, y in enumerate(mainclass.years):
            #     seriesr = xrsub.sel(Time=y).Value
            #     var_totalraw[y_i] = np.nanvar(seriesr/np.mean(seriesr))
            return xrsub
        
        def generate_listoflists(self, var, mainclass, xrsub):
            xrmeta = mainclass.meta.XR
            values = np.array(xrsub.Value)
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

        def generate_samples(self, var, mainclass, xrsub):
            xrmeta = mainclass.meta.XR
            values = np.array(xrsub.Value)
            values_nn = np.array(xrsub.Value)
            values_nn = values_nn / np.mean(values_nn)
            values = values - np.mean(values)
            values = values / np.std(values)
            mainclass.values = values
            modscens = np.array(xrsub.ModelScenario)
            mods = np.array([i.split('|')[0] for i in modscens])
            scens = np.array([i.split('|')[1] for i in modscens])
            ccat = np.array(xrmeta.sel(ModelScenario=xrsub.ModelScenario).Category.data)
            unimods = np.unique(mods)
            uniccat = np.unique(ccat)
            uniscen = np.unique(scens)

            # Generate samples
            whs = self.generate_listoflists(var, mainclass, xrsub)
            ss = mainclass.sample_size_per_ms

            indices = np.zeros(shape=(6, mainclass.resampling, len(mainclass.years)))
            for n_i in range(mainclass.resampling):
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

                M1 = np.zeros(shape=(len(sample1[0]), len(mainclass.years)))
                M1nn = np.zeros(shape=(len(sample1[0]), len(mainclass.years)))
                M2 = np.zeros(shape=(len(sample1[0]), len(mainclass.years)))
                Nm = np.zeros(shape=(len(sample1[0]), len(mainclass.years)))
                Nc = np.zeros(shape=(len(sample1[0]), len(mainclass.years)))
                Nmc = np.zeros(shape=(len(sample1[0]), len(mainclass.years)))
                for m in unimods:
                    for c in uniccat:
                        wh = np.where((mods == m) & (ccat == c))[0]
                        if len(wh) > 0:
                            wh1 = np.where((sample1[0] == m) & (sample1[1] == c))[0]
                            wh2 = np.where((sample2[0] == m) & (sample2[1] == c))[0]
                            choice = np.random.choice(wh, mainclass.sample_size_per_ms, replace=True)
                            M1nn[wh1] = values_nn[choice]
                            M1[wh1] = values[choice]
                            M2[wh2] = values[np.random.choice(wh, mainclass.sample_size_per_ms, replace=True)]

                            wh_m = np.where((sample1[0] == m) & (sample2[1] == c))[0]
                            wh_c = np.where((sample2[0] == m) & (sample1[1] == c))[0]
                            Nm[wh_m] = values[np.random.choice(wh, len(wh_m), replace=True)]
                            Nc[wh_c] = values[np.random.choice(wh, len(wh_c), replace=True)]

                            Nmc[wh1] = values[np.random.choice(wh, len(wh1), replace=True)]

                vtot = np.var(M1nn, axis=0)
                vtot_norm = np.var(M1, axis=0)
                s_m = np.diag(1/(len(sample1[0])-1)*np.dot(M1.T, Nm) - 1/(len(sample1[0]))*np.dot(M1.T, M2))/np.var(M1, axis=0)
                s_c = np.diag(1/(len(sample1[0])-1)*np.dot(M1.T, Nc) - 1/(len(sample1[0]))*np.dot(M1.T, M2))/np.var(M1, axis=0)
                comb = np.diag(1/(len(sample1[0])-1)*np.dot(M1.T, Nmc) - 1/len(sample1[0])*np.dot(M1.T, M2))/np.var(M1, axis=0)
                s_mc = comb - s_m - s_c
                s_z = 1 - comb
                indices[:, n_i, :] = [vtot, s_m, s_c, s_mc, s_z, vtot_norm]

            mainclass.comb = comb
            
            return np.mean(indices, axis=1)

        def decompose_variable(self, var, mainclass, xrsample):

            # Helper functions
            def first_order(column, series):
                array = np.copy(column)
                uniarray = np.unique(array)
                means = []
                for m in range(len(uniarray)):
                    wh=np.where(array == uniarray[m])[0]
                    means = means+[np.mean(series[wh])]*len(wh)
                return np.var(means)
            
            def combined_first_order(column1, column2, series):
                array1 = np.copy(column1)
                uniarray1 = np.unique(array1)
                array2 = np.copy(column2)
                uniarray2 = np.unique(array2)
                means = []
                for m in uniarray1:
                    wh1 = np.where(array1 == m)[0]
                    for n in uniarray2:
                        wh2 = wh1[array2[wh1] == n]
                        means = means+[np.mean(series[wh2])]*len(wh2)
                return np.var(means)
            
            # Start calculations
            s_z = np.zeros(shape=(len(mainclass.years)))+np.nan
            s_c = np.zeros(shape=(len(mainclass.years)))+np.nan
            s_m = np.zeros(shape=(len(mainclass.years)))+np.nan
            s_mc = np.zeros(shape=(len(mainclass.years)))+np.nan

            # Sample rows
            mods_sample = np.array([xrsample.ModelScenario.data[i].split('|')[0] for i in range(len(xrsample.ModelScenario))])
            ccat_sample = np.array(mainclass.meta.XR.sel(ModelScenario=xrsample.ModelScenario).Category.data)

            # Calculation of S_m, S_c, S_mc and S_z
            for y_i, y in enumerate(mainclass.years):
                seriesr = xrsample.sel(Time=y).Value
                series = seriesr - np.mean(seriesr)
                series = series / np.std(seriesr) # Var = 1 so no division needed anymore
                s_m[y_i] = first_order(mods_sample, series)
                s_c[y_i] = first_order(ccat_sample, series)
                comb = combined_first_order(mods_sample, ccat_sample, series)
                s_mc[y_i] = comb-s_m[y_i]-s_c[y_i]
                s_z[y_i] = 1-comb

            # Return
            return np.array([s_m, s_c, s_z, s_mc], dtype=object)

# =========================================================== #
# INITIALIZATION OF CLASS WHEN CALLED
# =========================================================== #

if __name__ == "__main__":

    vardec = variancedecomp()
    vardec.generate_modscencounts()

# =========================================================== #
# OLD CODE
# =========================================================== #

            # for n_i in range(mainclass.times_sampling):
            #     # Decompose based on model
            #     sample = []
            #     for m in np.unique(mods):
            #         wh = np.where(mods == m)[0]
            #         sample = sample+list(np.random.choice(wh, mainclass.sample_size, replace=False))
            #     sample = modscen[np.array(sample)]
            #     xrsample = xrsub.sel(ModelScenario = sample)
            #     mods_sample = np.array([xrsample.ModelScenario.data[i].split('|')[0] for i in range(len(xrsample.ModelScenario))])
            #     for y_i, y in enumerate(years):
            #         seriesr = xrsample.sel(Time=y).Value
            #         series = seriesr - np.mean(seriesr)
            #         series = series / np.std(seriesr)
            #         var_model[n_i, y_i] = first_order(mods_sample, series)#/np.nanvar(series)

            #     # Decompose based on ccat
            #     sample = []
            #     for c in np.unique(ccat):
            #         wh = np.where(ccat == c)[0]
            #         sample = sample+list(np.random.choice(wh, mainclass.sample_size, replace=False))
            #     sample = modscen[np.array(sample)]
            #     xrsample = xrsub.sel(ModelScenario = sample)
            #     ccat_sample = np.array(mainclass.meta.XR.sel(ModelScenario=xrsample.ModelScenario).Category.data)
            #     for y_i, y in enumerate(years):
            #         seriesr = xrsample.sel(Time=y).Value
            #         series = seriesr - np.mean(seriesr)
            #         series = series / np.std(seriesr)
            #         var_ccat[n_i, y_i] = first_order(ccat_sample, series)#/np.nanvar(series)

            #     # Decompose based on both model and ccat

            #     sample = []
            #     for c in np.unique(ccat):
            #         for m in np.unique(mods):
            #             wh = np.where((ccat == c) & (mods == m))[0]
            #             if len(wh) >= mainclass.combined_sample_size:
            #                 sample = sample+list(np.random.choice(wh, mainclass.combined_sample_size))#list(np.random.choice(wh, max([1, int(0.2*mainclass.sample_size)])))
            #     sample = modscen[np.array(sample)]
            #     xrsample = xrsub.sel(ModelScenario = sample)
            #     mods_sample = np.array([xrsample.ModelScenario.data[i].split('|')[0] for i in range(len(xrsample.ModelScenario))])
            #     ccat_sample = np.array(mainclass.meta.XR.sel(ModelScenario=xrsample.ModelScenario).Category.data)
            #     for y_i, y in enumerate(years):
            #         seriesr = xrsample.sel(Time=y).Value
            #         series = seriesr - np.mean(seriesr)
            #         series = series / np.std(seriesr)
            #         var_other[n_i, y_i] = 1-combined_first_order(mods_sample, ccat_sample, series)#/np.nanvar(series)
