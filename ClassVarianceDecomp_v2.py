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
    # =========================================================== #

    def __init__(self):
        print("# ==================================== #")
        print("# Initializing variancedecomp class    #")
        print("# ==================================== #")

        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            diction = yaml.load(file, Loader=yaml.FullLoader)
        self.location_ipccdata = Path(diction['location_ipcc'])
        self.varfile = diction['varfile']
        self.save = diction['save']
        self.threshold_dataremoval = int(diction['threshold_dataremoval'])
        self.sample_size_per_ms = int(diction['sample_size_per_ms'])
        self.variable_counting = int(diction['variable_counting'])
        self.resampling = int(diction['resampling'])
        self.removal_c8 = diction['removal_c8']
        self.generate_composites = diction['generate_composites']

        # Paths
        self.path_meta = self.location_ipccdata / "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
        self.path_var = self.current_dir / "Data" / self.varfile
        self.path_ar6 = self.location_ipccdata / "AR6_Scenarios_Database_World_v1.1.csv"
        self.path_modelconv = self.location_ipccdata / "ModelConversion.xlsx"

        #print('- Applying sampled decomposition')
        #self.decomp = variancedecomp.class_decomposition(self)
        #print('- Done!')

    # =========================================================== #
    # =========================================================== #

    def read_variable_list(self):
        print('- Reading variable list')
        xls = pd.read_excel(self.path_var, sheet_name = "Data")
        vars = np.array(xls["Variable"])
        cats = np.array(xls["Category"])
        fract = np.array(xls["Fractional"])
        gen = np.array(xls["Generator"]).astype(str)
        self.var_categories = np.unique(cats)
        self.var_raw = vars
        self.var_gen = gen
        self.var_unigen = np.unique(gen[gen != 'nan'])
        self.var_listed = []
        for c in self.var_categories:
            self.var_listed.append(list(vars[cats == c]))
        self.var_all = np.unique(list(self.var_raw)+list(self.var_unigen))
        self.var_fract = fract.astype(str)
        self.to_fractionize = self.var_raw[self.var_fract != 'nan']

    # =========================================================== #
    # =========================================================== #
    
    def read_metadata(self):
        print('- Reading metadata')
        DF = pd.read_excel(self.path_meta, sheet_name='meta_Ch3vetted_withclimate')
        idx = np.where((DF.Scenario == 'EN_NPi2020_800') & (DF.Model == 'WITCH 5.0'))[0]
        DF = DF.drop(idx)
        DF = DF.reset_index(drop=True)
        idx = np.where((DF.Scenario == 'EN_NPi2020_900') & (DF.Model == 'WITCH 5.0'))[0]
        DF = DF.drop(idx)
        DF = DF.reset_index(drop=True)
        DF = DF[DF.Vetting_historical == 'Pass']
        if self.removal_c8 == "no":
            DF.Category[DF.Category == "C8"] = "C7-8"
            DF.Category[DF.Category == "C7"] = "C7-8"
        self.DFmeta = DF.reset_index(drop=True)
        self.model = np.array(DF.Model)
        self.scen = np.array(DF.Scenario)
        self.modscen = np.array([self.model[i]+'|'+self.scen[i] for i in range(len(DF))])

    # =========================================================== #
    # =========================================================== #
    
    def read_ar6(self):
        print('- Reading AR6 data')
        DF = pd.read_csv(self.path_ar6)
        Mods_raw = np.array(DF.Model)
        Scen_raw = np.array(DF.Scenario)
        DF['ModelScenario'] = [Mods_raw[i]+'|'+Scen_raw[i] for i in range(len(Scen_raw))]
        DF = DF[DF.ModelScenario.isin(self.modscen)]
        DF = DF.reset_index()[DF.keys()]
        self.DFar6 = DF

    # =========================================================== #
    # =========================================================== #
    
    def count_variables(self):
        print('- Count variables in total database')
        Mods_raw = np.array(self.DFar6.Model)
        Scen_raw = np.array(self.DFar6.Scenario)
        Vars_raw = np.array(self.DFar6.Variable)
        modscen = np.array([Mods_raw[i]+'|'+Scen_raw[i] for i in range(len(Scen_raw))])
        d = {}
        d['Variable'] = list(np.unique(Vars_raw))
        count = []
        for i in np.unique(Vars_raw):
            count.append(len(np.unique(modscen[Vars_raw == i])))
        d['Count'] = count
        self.countsdf = pd.DataFrame(d)

    # =========================================================== #
    # =========================================================== #
    
    def filters_and_preprocessing(self):
        print("- Applying various filters and preprocessing to the database")
        DF = self.DFar6[self.DFar6.Variable.isin(self.var_raw)]
        DF = DF.reset_index(drop=True)
        DF = DF.drop(columns=['Region', 'Unit'])
        DF = DF.drop(columns=['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002',
                                '2003', '2004', '2005', '2006', '2007', '2008', '2009'])
        Mods_raw = np.array(DF.Model)
        Scens = np.array(DF.Scenario)
        ModScen_raw = [Mods_raw[i]+'|'+Scens[i] for i in range(len(DF))]
        wh = []
        for i in range(len(ModScen_raw)):
            if ModScen_raw[i] in self.modscen:
                wh.append(i)
        DF = DF[DF.index.isin(wh)]
        DF = DF.reset_index(drop=True)
        self.DFar6 = DF
        self.model = np.array(DF.Model)
        self.scen = np.array(DF.Scenario)
        self.vars = np.array(DF.Variable)

    # =========================================================== #
    # =========================================================== #
    
    def merge_model_versions_and_scenarios(self):
        print("- Merge model versions and combine with scenario names")
        self.modelversionsdf = pd.read_excel(self.path_modelconv, sheet_name='Sheet1')
        modelversions_model = np.array(self.modelversionsdf['Model'])
        metamodel = np.array(self.DFmeta.Model)
        metascen = np.array(self.DFmeta.Scenario)
        for v_i, vers in enumerate(self.modelversionsdf['Model version']):
            mod = modelversions_model[v_i]
            metamodel[metamodel == vers] = mod
            self.model[self.model == vers] = mod
        self.DFmeta['model'] = metamodel
        self.DFar6['Model'] = self.model
        self.DFmeta['ModelScenario'] = [metamodel[i]+'|'+metascen[i] for i in range(len(metascen))]
        self.DFmeta = self.DFmeta.drop(columns=['Model', 'Scenario'])
        self.modscen = np.array([self.model[i]+'|'+self.scen[i] for i in range(len(self.scen))])
        self.DFar6['ModelScenario'] = self.modscen
        self.DFar6 = self.DFar6[['ModelScenario', 'Variable']+list(np.arange(2010, 2101).astype(str))]

    # =========================================================== #
    # =========================================================== #
    
    def generate_aggregated_variables(self):
        print("- Generate aggregated variables")
        DSs = []
        for gen in self.var_unigen:
            whichvar = self.var_raw[self.var_gen == gen]
            unims = np.unique(self.DFar6.ModelScenario)
            df_vars = np.array(self.DFar6.Variable)
            df_ms = np.array(self.DFar6.ModelScenario)
            df_data = np.array(self.DFar6[list(np.arange(2010, 2101).astype(str))])

            newrows = []
            whichms = []
            for ms in tqdm(unims):
                newdat = np.zeros(shape=(len(whichvar), 91))+np.nan
                checker = 0
                for v_i, v in enumerate(whichvar):
                    dat = df_data[(df_vars == v) & (df_ms == ms)]
                    if len(dat) != 0:
                        newdat[v_i] = dat[0]
                if np.nansum(newdat) > 0:
                    whichms.append(ms)
                    newdata = np.nansum(np.array(newdat), axis=0)
                    newdata[newdata == 0] = np.nan
                    newrows.append(newdata)
            newrows = np.array(newrows)
            dic = {"ModelScenario": whichms, "Variable": gen}
            dic.update({str(y) : newrows[:, y_i] for y_i, y in enumerate(range(2010, 2101))})
            DS = pd.DataFrame(dic)
            DSs.append(DS)
        DFar6 = pd.concat([self.DFar6, DSs[0], DSs[1], DSs[2], DSs[3]], ignore_index=True)
        self.DFar6 = DFar6

    # =========================================================== #
    # =========================================================== #
    
    def add_fractional_variables(self):
        print("- Add fractional variables")
        vars = np.array(self.DFar6.Variable)
        modscen = np.array(self.DFar6.ModelScenario)
        arrays = []
        for v_i, v in enumerate(vars):
            if v in self.to_fractionize:
                f = self.var_fract[np.where(self.var_raw == v)[0][0]]
                newdf = []
                newdf.append(modscen[v_i])
                newdf.append(v+" (fr)")
                try:
                    data = np.array(self.DFar6[(vars == v) & (modscen == modscen[v_i])])[0, 2:]/np.array(self.DFar6[(vars == f) & (modscen == modscen[v_i])])[0, 2:]
                    for y in range(len(data)):
                        newdf.append(data[y])
                    arrays.append(newdf)
                except:
                    continue
        newdf = pd.DataFrame(arrays, columns = self.DFar6.keys())
        self.DFar6 = pd.concat([self.DFar6, newdf], ignore_index=True)
        self.var_all = np.unique(self.DFar6.Variable)
        # vardf = np.array(self.DFar6.Variable)
        # modscen = np.array(self.DFar6.ModelScenario)
        # unimodscen = np.unique(modscen)
        # unifract = np.unique(self.var_fract)
        # unifract = unifract[unifract != 'nan']
        # for f_i, f in enumerate(unifract):
        #     vars = self.var_raw[np.where(self.var_fract == f)[0]]
        #     if f_i == 0:
        #         obj = self.XRar6
        #     else:
        #         obj = xrfr
        #     if f == 'Final Energy|Industry': xrfr = obj.assign(Ind_f = self.XRar6.sel(Variable = vars).Value/self.XRar6.sel(Variable = f).Value)
        #     if f == 'Final Energy|Transportation': xrfr = obj.assign(Transp_f = self.XRar6.sel(Variable = vars).Value/self.XRar6.sel(Variable = f).Value)
        #     if f == 'Final Energy|Residential and Commercial': xrfr = obj.assign(Res_f = self.XRar6.sel(Variable = vars).Value/self.XRar6.sel(Variable = f).Value)
        #     if f == 'Primary Energy': xrfr = obj.assign(PE_f = self.XRar6.sel(Variable = vars).Value/self.XRar6.sel(Variable = f).Value)
        #     if f == 'Secondary Energy|Electricity': xrfr = obj.assign(Elec_f = self.XRar6.sel(Variable = vars).Value/self.XRar6.sel(Variable = f).Value)
        #     if f == 'Secondary Energy|Hydrogen': xrfr = obj.assign(H2_f = self.XRar6.sel(Variable = vars).Value/self.XRar6.sel(Variable = f).Value)
        #     if f == 'Secondary Energy|Liquids': xrfr = obj.assign(Liq_f = self.XRar6.sel(Variable = vars).Value/self.XRar6.sel(Variable = f).Value)
        # self.XRar6 = xrfr

    # =========================================================== #
    # =========================================================== #
    
    def conversion_to_xarray(self):
        print("- Convert time dimension and xarray")
        XRdummy = self.DFmeta.set_index(['ModelScenario'])
        self.XRmeta = xr.Dataset.from_dataframe(XRdummy)
        DF_timadj = self.DFar6.melt(id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value")
        DF_timadj['Time'] = np.array(DF_timadj['Time']).astype(int)
        dfdummy = DF_timadj.set_index(['ModelScenario', 'Variable', 'Time'])
        self.XRar6 = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #
    
    def temporal_interpolation(self):
        print("- Temporal interpolation")
        self.XRar6 = self.XRar6.sel(Variable=self.var_all)
        self.XRar6 = self.XRar6.interpolate_na(dim="Time", method="linear")
        self.years = list(self.XRar6.Time.data)

    # =========================================================== #
    # =========================================================== #
    
    def count_models_and_climate_outcomes(self):
        print("- Count models and c-outcomes")
        # Model counts for each mod-scen pair
        Models = [self.XRar6.ModelScenario.data[i].split('|')[0] for i in range(len(self.XRar6.ModelScenario))]
        DFmods = {}
        unimod = np.unique(Models)
        DFmods['Model'] = ['Total']+list(unimod)
        for v in range(len(self.var_listed)+1):
            numscens = []
            for m in unimod:
                XR_sub = self.XRar6.sel(Variable=([self.var_all]+self.var_listed)[v])
                XR_sub = XR_sub.dropna('ModelScenario', how='any')
                mods = np.array([XR_sub.ModelScenario.data[i].split('|')[0] for i in range(len(XR_sub.ModelScenario))]).astype(str)
                numscens.append(len(np.where(mods == m)[0]))
            if v == 0:
                DFmods['Full set'] = [sum(numscens)]+numscens
            else:
                DFmods[self.var_categories[v-1]] = [sum(numscens)]+numscens
        DFmods = pd.DataFrame(DFmods)
        DFmods = DFmods.sort_values(by=['Full set'], ascending=False)
        DFmods = DFmods.reset_index(drop=True)
        self.count_dfm = DFmods
        
        # C-category counts for each mod-scen pair
        Ccategories = np.array(self.XRmeta.sel(ModelScenario=self.XRar6.ModelScenario).Category.data)
        DFc = {}
        unicat = np.unique(Ccategories)
        DFc['Ccategory'] = ['Total']+list(unicat)
        for v in range(len(self.var_listed)+1):
            numscens = []
            for m in unicat:
                XR_sub = self.XRar6.sel(Variable=([self.var_all]+self.var_listed)[v])
                XR_sub = XR_sub.dropna('ModelScenario', how='any')
                ccat = np.array(self.XRmeta.sel(ModelScenario=XR_sub.ModelScenario).Category.data)
                numscens.append(len(np.where(ccat == m)[0]))
            if v == 0:
                DFc['Full set'] = [sum(numscens)]+numscens
            else:
                DFc[self.var_categories[v-1]] = [sum(numscens)]+numscens
        DFc = pd.DataFrame(DFc)
        DFc = DFc.reset_index(drop=True)
        self.count_dfc = DFc

    # =========================================================== #
    # =========================================================== #
    
    def filters(self):
        print("- Filter out scenarios with C8 and models with small entry numbers")
        self.XRsubs = {}
        for var in self.var_all:
            modscen_new = []
            xrsub = self.XRar6.sel(Variable=var)
            xrsub = xrsub.dropna('ModelScenario', how='any')
            modscen = np.array(xrsub.ModelScenario)
            mods = np.array([xrsub.ModelScenario.data[i].split('|')[0] for i in range(len(xrsub.ModelScenario))])
            ccat = np.array(self.XRmeta.sel(ModelScenario=xrsub.ModelScenario).Category.data)
            if self.removal_c8 == "yes":
                for i in range(len(mods)):
                    m = mods[i]
                    c = ccat[i]
                    if len(np.where(mods == m)[0]) >= self.threshold_dataremoval and c != 'C8':
                        modscen_new.append(modscen[i])
            elif self.removal_c8 == "no":
                for i in range(len(mods)):
                    m = mods[i]
                    c = ccat[i]
                    if len(np.where(mods == m)[0]) >= self.threshold_dataremoval:
                        modscen_new.append(modscen[i])
            xrsub = xrsub.sel(ModelScenario = modscen_new)
            self.XRsubs[var] = xrsub

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
        st = ""
        if self.removal_c8 == "no":
            st = "_c78"
        #st = "_lowres"
        if self.save == 'yes':
            try:
                self.countsdf.to_csv(self.current_dir / "Data" / ("Counts"+st+".csv"))
            except:
                3
            self.XRmeta[["ModelScenario", "Category", "IMP_marker", "Policy_category"]].to_netcdf(self.current_dir / "Data" / ("XRmeta"+st+".nc"))
            self.XRar6.to_netcdf(self.current_dir / "Data" / ("XRdata"+st+".nc"))
            self.count_dfc.to_csv(self.current_dir / "Data" / ("Ccategories"+st+".csv"))
            self.count_dfm.to_csv(self.current_dir / "Data" / ("Models"+st+".csv"))
            self.XRvar.to_netcdf(self.current_dir / "Data" / ("Variances"+st+".nc"))

# =========================================================== #
# INITIALIZATION OF CLASS WHEN CALLED
# =========================================================== #

if __name__ == "__main__":

    vardec = variancedecomp()
    vardec.generate_modscencounts()