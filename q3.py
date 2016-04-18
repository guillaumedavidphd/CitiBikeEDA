get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import zipfile as z
import requests
import io
import matplotlib as mpl
import matplotlib.pyplot as plt

from bokeh.io import hplot, vform, output_notebook, save
from bokeh.models import CustomJS, ColumnDataSource, Slider, Range1d
from bokeh.plotting import Figure, show

from bokeh.tile_providers import STAMEN_TONER
output_notebook()

import seaborn as sns
sns.set()
mpl.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (20, 20)


def convLL_XY(lat, lon):
    mapWidth    = 2*20037508.34
    mapHeight   = 2*20037508.34

    x = (lon+180)*(mapWidth/360)

    latRad = lat*np.pi/180
    mercN = np.log(np.tan((np.pi/4)+(latRad/2)))
    y     = (mapHeight/2)-(mapWidth*mercN/(2*np.pi))

    return (x-mapWidth/2, y-mapHeight/2)

def convLL_XY_Df(df):
    mapWidth    = 2*20037508.34
    mapHeight   = 2*20037508.34

    df['x'] = (df.iloc[:, 2]+180)*(mapWidth/360)

    latRad = -df.iloc[:, 1]*np.pi/180
    mercN = np.log(np.tan((np.pi/4)+(latRad/2)))
    df['y']     = (mapHeight/2)-(mapWidth*mercN/(2*np.pi))

    df['x'] = df['x']-mapWidth/2
    df['y'] = df['y']-mapWidth/2

    return df


def plotMap(lat, lon, axis_range):
    x_offset, y_offset = convLL_XY(lat, lon)
    x_axis_range = [-axis_range+x_offset, axis_range+x_offset]
    y_axis_range = [-axis_range+y_offset, axis_range+y_offset]
    x_range = Range1d(start=x_axis_range[0], end=x_axis_range[1])
    y_range = Range1d(start=y_axis_range[0], end=y_axis_range[1])
    p = Figure(tools='pan,wheel_zoom', x_range=x_range, y_range=y_range, plot_height=800, plot_width=800)
    p.axis.visible = False
    STAMEN_TONER.url = 'http://tile.stamen.com/toner-lite/{Z}/{X}/{Y}.png'
    p.add_tile(STAMEN_TONER)
    return p


years = np.arange(2016, 2017)
# years = np.arange(2015, 2016) # uncomment to get 2015 data
months = np.arange(2, 3)
# months = np.arange(1, 13) # uncomment to get 2015 data
df = pd.DataFrame()
for year in years:
    for month in months:
        fn = "%d"%year+"%02d"%month+"-citibike-tripdata.zip"
        csvname = fn[0:-3]+'csv'
        r = requests.get('https://s3.amazonaws.com/tripdata/'+fn)
        zipfile = z.ZipFile(io.BytesIO(r.content))
        df = pd.concat([df, pd.read_csv(zipfile.open(csvname), parse_dates=[1, 2], infer_datetime_format=True)])
df = df.reset_index()
df.drop('index', axis=1, inplace=True)

lut_Start = pd.DataFrame(df[['start station name',
                                    'start station latitude',
                                    'start station longitude']],
                                index=df['start station id'])
lut_Start = lut_Start.groupby(level=0)
lut_Start = lut_Start.first()
lut_Start = convLL_XY_Df(lut_Start)

lut_End = pd.DataFrame(df[['end station name',
                                         'end station latitude',
                                         'end station longitude']],
                                     index=df['end station id'])
lut_End = lut_End.groupby(level=0)
lut_End = lut_End.first()
lut_End = convLL_XY_Df(lut_End)

df = df.drop(['start station name', 'end station name'], axis=1)
df['usertype'] = df['usertype'].apply(lambda x: 1 if x=='Subscriber' else 0)
df['gender'] = df['gender'].apply(lambda x: 1 if x==0 else x)
dfBirthYearMedian = df['birth year'].median()
df['birth year'] = df['birth year'].apply(lambda x: dfBirthYearMedian if pd.isnull(x) else x)

df['usertype'].describe()

df_subs = df[df['usertype']==1]

sns.distplot(df_subs['birth year'], kde=False, bins=20, color='#3182bd')

df_birthyear = df_subs['birth year']
df_birthyear = df_birthyear[df_birthyear>=1915]

df_birthyear.describe()

df_subs['gender'].hist(color='#3182bd')

(df_subs['gender']-1).describe()

grp = df_subs.groupby('gender')['tripduration', 'birth year'].mean()

grp

df_StartCount = df.loc[:, ['start station id', 'starttime']]
df_StartCount['started trips'] = 1

df_StartCount = df_StartCount.groupby([pd.Grouper(freq='H', key='starttime'), 'start station id']).sum()
df_StartCount = df_StartCount.unstack(level=0)['started trips'].reset_index()
df_StartCount = pd.merge(lut_Start.reset_index(), df_StartCount.fillna(0))
df_StartCount['radius'] = df_StartCount.loc[:, pd.to_datetime(df_StartCount.columns[6])]
df_StartCount.columns = df_StartCount.columns.format()

radius_factor = 20

start_day = '20160201'
stop_day = '20160202'
source1 = ColumnDataSource(
    radius_factor*pd.concat([df_StartCount.loc[:, pd.to_datetime(start_day).strftime('%Y-%m-%d %H:%M:%S'):pd.to_datetime(stop_day).strftime('%Y-%m-%d %H:%M:%S')],
               df_StartCount.loc[:, 'radius']],
               axis=1)
)

lat = -40.7437815
lon = -73.9757094
axis_range = 5e3

plot1 = plotMap(lat, lon, axis_range)
plot1.circle(x=df_StartCount['x'].values,
            y=df_StartCount['y'].values,
            radius='radius',
            source=source1,
            color='#3186cc',
            fill_color='#3186cc',
            alpha=.4)
plot1.title='Hourly volume of trips started in 1 day'

df_EndCount = df.loc[:, ['end station id', 'stoptime']]
df_EndCount['ended trips'] = 1

df_EndCount = df_EndCount.groupby([pd.Grouper(freq='H', key='stoptime'), 'end station id']).sum()
df_EndCount = df_EndCount.unstack(level=0)['ended trips'].reset_index()
df_EndCount = pd.merge(lut_End.reset_index(), df_EndCount.fillna(0))
df_EndCount['radius'] = df_EndCount.loc[:, pd.to_datetime(df_EndCount.columns[6])]
df_EndCount.columns = df_EndCount.columns.format()

source2 = ColumnDataSource(
    radius_factor*pd.concat([df_EndCount.loc[:, pd.to_datetime(start_day).strftime('%Y-%m-%d %H:%M:%S'):pd.to_datetime(stop_day).strftime('%Y-%m-%d %H:%M:%S')],
               df_EndCount.loc[:, 'radius']],
               axis=1)
)

plot2 = plotMap(lat, lon, axis_range)
plot2.circle(x=df_EndCount['x'].values,
            y=df_EndCount['y'].values,
            radius='radius',
            source=source2,
            color='#de2d26',
            fill_color='#de2d26',
            alpha=.4)
plot2.x_range = plot1.x_range
plot2.y_range = plot1.y_range
plot2.title='Hourly volume of trips ended in 1 day'

def callback(source1=source1, source2=source2):
    data1 = source1.get('data')
    f = cb_obj.get('value')
    date = sorted(data1.keys())[f-1]
    r = data1[date]
    data1['radius'] = r
    source1.trigger('change')
    data2 = source2.get('data')
    r = data2[date]
    data2['radius'] = r
    source2.trigger('change')

slider = Slider(start=1, end=len(sorted(source1.data.keys())[:-4]), value=1, step=1, title="hour",
                callback=CustomJS.from_py_func(callback),
                orientation='horizontal')

layout = hplot(plot1, plot2)
layout = vform(layout, slider)

show(layout)
save(layout, 'HourlyTripsOneDay.html')

df_StartCount = df.loc[:, ['start station id', 'starttime']]
df_StartCount['started trips'] = 1

df_StartCount = df_StartCount.groupby([pd.Grouper(freq='D', key='starttime'), 'start station id']).sum()
df_StartCount = df_StartCount.unstack(level=0)['started trips'].reset_index()
df_StartCount = pd.merge(lut_Start.reset_index(), df_StartCount.fillna(0))
df_StartCount['radius'] = df_StartCount.loc[:, pd.to_datetime(df_StartCount.columns[6])]
df_StartCount.columns = df_StartCount.columns.format()

radius_factor = 3

start_day = '20160201'
stop_day = '20160207'
source1 = ColumnDataSource(
    radius_factor*pd.concat([df_StartCount.loc[:, pd.to_datetime(start_day).strftime('%Y-%m-%d %H:%M:%S'):pd.to_datetime(stop_day).strftime('%Y-%m-%d %H:%M:%S')],
               df_StartCount.loc[:, 'radius']],
               axis=1)
)

lat = -40.7437815
lon = -73.9757094
axis_range = 5e3

plot1 = plotMap(lat, lon, axis_range)
plot1.circle(x=df_StartCount['x'].values,
            y=df_StartCount['y'].values,
            radius='radius',
            source=source1,
            color='#3186cc',
            fill_color='#3186cc',
            alpha=.4)
plot1.title='Daily volume of trips started in 1 week'

df_EndCount = df.loc[:, ['end station id', 'stoptime']]
df_EndCount['ended trips'] = 1

df_EndCount = df_EndCount.groupby([pd.Grouper(freq='D', key='stoptime'), 'end station id']).sum()
df_EndCount = df_EndCount.unstack(level=0)['ended trips'].reset_index()
df_EndCount = pd.merge(lut_End.reset_index(), df_EndCount.fillna(0))
df_EndCount['radius'] = df_EndCount.loc[:, pd.to_datetime(df_EndCount.columns[6])]
df_EndCount.columns = df_EndCount.columns.format()

source2 = ColumnDataSource(
    radius_factor*pd.concat([df_EndCount.loc[:, pd.to_datetime(start_day).strftime('%Y-%m-%d %H:%M:%S'):pd.to_datetime(stop_day).strftime('%Y-%m-%d %H:%M:%S')],
               df_EndCount.loc[:, 'radius']],
               axis=1)
)

plot2 = plotMap(lat, lon, axis_range)
plot2.circle(x=df_EndCount['x'].values,
            y=df_EndCount['y'].values,
            radius='radius',
            source=source2,
            color='#de2d26',
            fill_color='#de2d26',
            alpha=.4)
plot2.x_range = plot1.x_range
plot2.y_range = plot1.y_range
plot2.title='Daily volume of trips ended in 1 week'

def callback(source1=source1, source2=source2):
    data1 = source1.get('data')
    f = cb_obj.get('value')
    date = sorted(data1.keys())[f-1]
    r = data1[date]
    data1['radius'] = r
    source1.trigger('change')
    data2 = source2.get('data')
    r = data2[date]
    data2['radius'] = r
    source2.trigger('change')

slider = Slider(start=1, end=len(sorted(source1.data.keys())[:-4]), value=1, step=1, title="day",
                callback=CustomJS.from_py_func(callback),
                orientation='horizontal')

layout = hplot(plot1, plot2)
layout = vform(layout, slider)

show(layout)
save(layout, 'DailyTripsOneWeek.html')
