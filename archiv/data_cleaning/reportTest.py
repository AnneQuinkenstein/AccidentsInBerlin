from ydata_profiling import ProfileReport
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
unfaelle = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo-mitLockdown_mitCorona.csv', sep=';')
unfaelle.head(10)
#unfaelle = scaler.fit_transform(unfaelle)
profile = ProfileReport(unfaelle, title='Unfaelle ReportTest', explorative=True)

profile.to_file("Datensatz_Test.html")

