# ======================================== #
# Class that does the data handling
# Input: AR6 database v1.1
# Output: netcdf files that are read in by class_decomposition.py
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import xarray as xr
import yaml

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class data_handling(object):

    # =========================================================== #
    # =========================================================== #

    def __init__(self):
        print("# ==================================== #")
        print("# Initializing data_handling class     #")
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
        self.resampling = int(diction['resampling'])
        self.removal_c8 = diction['removal_c8']
        self.generate_composites = diction['generate_composites']

        # Paths
        self.path_meta = self.current_dir / self.location_ipccdata / "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
        self.path_var = self.current_dir / "Data" / "Input_files" / self.varfile
        self.path_ar6 = self.location_ipccdata / "AR6_Scenarios_Database_World_v1.1.csv"
        self.path_modelconv = self.location_ipccdata / "ModelConversion.xlsx"

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
    
    def savings(self):
        print("- Save stuff")
        if self.save == 'yes':
            self.XRmeta[["ModelScenario", "Category", "IMP_marker", "Policy_category"]].to_netcdf(self.current_dir / "Data" / "Handling_files" / "XRmeta.nc")
            self.XRar6.to_netcdf(self.current_dir / "Data" / "Handling_files" / ("XRdata.nc"))
            with open(self.current_dir / "Data" / "Handling_files" / 'XRsubs.pickle', 'wb') as handle:
                pickle.dump(self.XRsubs, handle, protocol=pickle.HIGHEST_PROTOCOL)
