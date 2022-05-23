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
        print('- Reading variable lists')
        self.vars = variancedecomp.class_vars(self)
        print('- Reading metadata')
        self.meta = variancedecomp.class_meta(self)
        print('- Reading AR6 data')
        self.ar6 = variancedecomp.class_ar6(self)
        print('- Done!')

    # =========================================================== #
    # Subclass variable lists (input)
    # =========================================================== #

    class class_vars(object):
        def __init__(self, mainclass):
            self.path = mainclass.current_dir / "Data" / "Variablelists.xlsx"
            xls = pd.read_excel(self.path, sheet_name = None)
            Names = list(xls.keys())
            Vlists = []
            FullVarlist = []
            for i in range(len(Names)):
                Vlists.append(list(pd.read_excel(self.path, sheet_name = Names[i]).Variable))
                FullVarlist = FullVarlist+list(pd.read_excel(self.path, sheet_name = Names[i]).Variable)
            self.full = FullVarlist+['Primary Energy']
            self.listed = Vlists
            self.listnames = Names

    # =========================================================== #
    # Subclass meta data
    # =========================================================== #
    
    class class_meta(object):
        def __init__(self, mainclass):
            self.path = mainclass.location_ipccdata / "ar6_full_metadata_indicators2021_10_14_v3.xlsx"
            DF = pd.read_excel(self.path, sheet_name='meta_Ch3vetted_withclimate')
            DF = DF[DF.Vetting_historical == 'PASS']
            self.DF = DF.reset_index(drop=True)
            self.model = np.array(DF.model)
            self.scen = np.array(DF.scenario)
            self.modscen = np.array([self.model[i]+'|'+self.scen[i] for i in range(len(DF))])

    # =========================================================== #
    # Subclass actual AR6 data
    # =========================================================== #
    
    class class_ar6(object):
        def __init__(self, mainclass):
            self.path = mainclass.location_ipccdata / "snapshot_world_with_key_climate_iamc_ar6_2021_10_14.csv"

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
            self.countsdf.to_csv(mainclass.current_dir / "Data" / "Counts.csv")

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
            mainclass.meta.DF = mainclass.meta.DF.drop(columns=['model', 'scenario'])
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
                if a==0:
                    DS = pd.DataFrame(d)
                else:
                    DS = DS.append(pd.DataFrame(d), ignore_index=True)
                a+=1
            self.DF = self.DF.append(DS)
            self.DF = self.DF.reset_index()
            self.DF = self.DF[self.DF.keys()[1:]]

            # ================================================================================ #
            print("     Convert time dimension and Xarray")
            # ================================================================================ #

            XRdummy = mainclass.meta.DF.set_index(['ModelScenario'])
            mainclass.meta.XR = xr.Dataset.from_dataframe(XRdummy)
            mainclass.meta.XR.to_netcdf(mainclass.current_dir / "Data" / "XRmeta.nc")

            DF_timadj = self.DF.melt(id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value")
            DF_timadj['Time'] = np.array(DF_timadj['Time']).astype(int)
            dfdummy = DF_timadj.set_index(['ModelScenario', 'Variable', 'Time'])
            self.XR = xr.Dataset.from_dataframe(dfdummy)

            # ================================================================================ #
            print("     Temporal interpolation")
            # ================================================================================ #

            self.XR = self.XR.sel(Variable=mainclass.vars.full)
            self.XR = self.XR.interpolate_na(dim="Time", method="linear")
            self.XR.to_netcdf(mainclass.current_dir / "Data" / "XRdata.nc")
    
    # =========================================================== #
    # Sublists for information (optional)
    # =========================================================== #
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
    
    class decomposition(object):
        def __init__(self, mainclass):
            3
        
        def draw(self):
            3

    def save(self):
        3
